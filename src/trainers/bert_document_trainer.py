import sys
import pdb
import os
import logging
import warnings
import math
import time
import shutil
import copy

import numpy as np
from packaging import version
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, LambdaLR
from torch.utils.data import DataLoader
from transformers import Trainer
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataset import Dataset

from tqdm.auto import tqdm

from transformers import __version__
from transformers.integrations import hp_params
from transformers.utils import (
    is_sagemaker_mp_enabled, is_apex_available, is_torch_tpu_available, is_datasets_available, CONFIG_NAME, WEIGHTS_NAME
)
from transformers.trainer_utils import (
    speed_metrics, has_length, denumpify_detensorize, EvalLoopOutput, EvalPrediction,
    set_seed, get_last_checkpoint, ShardedDDPOption, TrainOutput, HPSearchBackend, SchedulerType
)
from transformers.configuration_utils import PretrainedConfig
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.trainer_pt_utils import nested_detach, nested_numpify, nested_concat, nested_truncate, find_batch_size, IterableDatasetShard
from transformers.deepspeed import deepspeed_init, deepspeed_reinit
from transformers.optimization import Adafactor, TYPE_TO_SCHEDULER_FUNCTION, get_scheduler, get_polynomial_decay_schedule_with_warmup
from transformers.trainer_callback import TrainerState
from datasets import concatenate_datasets
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from utils.schedulers import OurReduceLROnPlateau, OurExponentialLR, get_multistep_lr_scheduler

if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import smp_forward_only, smp_nested_concat, smp_forward_backward

if is_apex_available():
    from apex import amp

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_datasets_available():
    import datasets

# Set PATH to SentEval
PATH_TO_SENTEVAL = '../SentEval'
PATH_TO_DATA = '../SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

# Import 
logger = logging.getLogger(__name__)

# Name of the files used for checkpointing
TRAINER_STATE_NAME = "trainer_state.json"


class BertTrainer(Trainer):

    def __init__(self, raw_dataset, model_args, data_args, eval_data_collator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        target_task = data_args.target_task
        self.target_tasks = target_task if isinstance(target_task, list) else [target_task]
        self.model_args = model_args
        self.data_args = data_args
        merged_dataset = concatenate_datasets([raw_dataset[split] for split in raw_dataset.keys()])
        self.dataset_to_sent_id = dict((sent, sent_id) for sent, sent_id in \
            zip(merged_dataset['text'], merged_dataset['sentences_ids']))
        # To help debug, reverse the dict
        self.id_to_sent = dict((sent_id, sent) for sent, sent_id in self.dataset_to_sent_id.items())
        self.eval_data_collator = eval_data_collator

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.
        Here we add OurReduceLROnPlateau Scheduler, OurExponentialLR, our_polynomial_decay_scheduler, and multistep scheduler.
        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:
            if self.model_args.use_special_lr_scheduler == 'reduceOnPlateau':
                self.lr_scheduler = OurReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, threshold_mode='abs', min_lr=5e-5)
            elif self.model_args.use_special_lr_scheduler == 'exponential':
                self.lr_scheduler = OurExponentialLR(optimizer, gamma=0.99, min_lr=0.0)
            elif self.model_args.use_special_lr_scheduler == 'polynomial':
                self.lr_scheduler = get_polynomial_decay_schedule_with_warmup(
                    optimizer=self.optimizer if optimizer is None else optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                    lr_end=5e-5,
                    power=3
                )
            elif self.model_args.use_special_lr_scheduler == 'multistep':
                self.lr_scheduler = get_multistep_lr_scheduler(
                    optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps
                )
            else:
                # Use Huggingface default lr_scheduler
                self.lr_scheduler = get_scheduler(
                    self.args.lr_scheduler_type,
                    optimizer=self.optimizer if optimizer is None else optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                )
        return self.lr_scheduler

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].
        Subclass and override this method if you want to inject some custom behavior.
        We add our own eval_data_collator for evaluation dataloader
        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is an `datasets.Dataset`, columns not accepted by
                the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        
        if isinstance(eval_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                eval_dataset = IterableDatasetShard(
                    eval_dataset,
                    batch_size=self.args.per_device_eval_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            return DataLoader(
                eval_dataset,
                batch_size=self.args.eval_batch_size,
                collate_fn=self.eval_data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            # We modified here!
            collate_fn=self.eval_data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Main training entry point.
        We modify the lr_scheduler.step() such that we can input metric like lr.scheduler.step(loss)
        We label all modifications with "Modified".
        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
            trial (`optuna.Trial` or `Dict[str, Any]`, *optional*):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            ignore_keys_for_eval (`List[str]`, *optional*)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
            kwargs:
                Additional keyword arguments used to hide deprecated arguments
        """
        resume_from_checkpoint = None if not resume_from_checkpoint else resume_from_checkpoint

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args = self.args

        self.is_in_train = True

        # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # the following is a workaround:
        if (args.fp16_full_eval or args.bf16_full_eval) and not args.do_train:
            self._move_model_to_device(self.model, args.device)

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        if resume_from_checkpoint is not None:
            if not os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
                raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

            logger.info(f"Loading model from {resume_from_checkpoint}).")

            if os.path.isfile(os.path.join(resume_from_checkpoint, CONFIG_NAME)):
                config = PretrainedConfig.from_json_file(os.path.join(resume_from_checkpoint, CONFIG_NAME))
                checkpoint_version = config.transformers_version
                if checkpoint_version is not None and checkpoint_version != __version__:
                    logger.warning(
                        f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                        f"Transformers but your current version is {__version__}. This is not recommended and could "
                        "yield to errors or unwanted behaviors."
                    )

            if args.deepspeed:
                # will be resumed in deepspeed_init
                pass
            else:
                # We load the model state dict on the CPU to avoid an OOM error.
                state_dict = torch.load(os.path.join(resume_from_checkpoint, WEIGHTS_NAME), map_location="cpu")
                # If the model is on the GPU, it still works!
                self._load_state_dict_in_model(state_dict)

                # release memory
                del state_dict

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self._move_model_to_device(self.model, args.device)
            self.model_wrapped = self.model

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                f"args.max_steps must be set to a positive value if dataloader does not have a length, was {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
            self.sharded_ddp is not None and self.sharded_ddp != ShardedDDPOption.SIMPLE or is_sagemaker_mp_enabled()
        )
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.
        
        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if version.parse(torch.__version__) < version.parse("1.11") or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                # Free the cuda memory periodically, MODIFIED!
                torch.cuda.empty_cache()
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                if (
                    ((step + 1) % args.gradient_accumulation_steps != 0)
                    and args.local_rank != -1
                    and args._no_sync_in_gradient_accumulation
                ):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss_step = self.training_step(model, inputs)
                else:
                    tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping

                        if self.do_grad_scaling:
                            # Reduce gradients first for XLA
                            if is_torch_tpu_available():
                                gradients = xm._fetch_gradients(self.optimizer)
                                xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                args.max_grad_norm,
                            )
                    
                    # Optimizer step
                    optimizer_was_run = True
                    if self.deepspeed:
                        pass  # called outside the loop
                    elif is_torch_tpu_available():
                        if self.do_grad_scaling:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            xm.optimizer_step(self.optimizer)
                    elif self.do_grad_scaling:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()
                    
                    if optimizer_was_run and not self.deepspeed:
                        # Modified Here!
                        # if using ReduceLROnPlateau, need to update it after evaluation
                        if self.model_args.use_special_lr_scheduler not in  ['reduceOnPlateau', 'exponential', 'multistep']:
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    f"There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            # Step the learning scheduler for epoch-level lr_scheduler
            # Modified Here!
            if self.model_args.use_special_lr_scheduler == 'reduceOnPlateau':
                # find the last evaluation result
                if len(self.state.log_history) > 0:
                    metric_name = self.args.metric_for_best_model
                    metric_name = 'eval_' + metric_name if 'eval_' not in metric_name else metric_name
                    last_metric = [history[metric_name] for history in self.state.log_history if metric_name in history][-1]
                else:
                    last_metric = 0.0
                self.lr_scheduler.step(last_metric)
            elif self.model_args.use_special_lr_scheduler == 'exponential':
                self.lr_scheduler.step()
            elif self.model_args.use_special_lr_scheduler == 'multistep':
                self.lr_scheduler.step()

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()

            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )

            best_model_path = os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)
            if os.path.exists(best_model_path):
                if self.deepspeed:
                    # temp hack until Deepspeed fixes the problem with resume from an existing engine that did some stepping
                    deepspeed_engine, optimizer, lr_scheduler = deepspeed_reinit(self)
                    self.model = deepspeed_engine.module
                    self.model_wrapped = deepspeed_engine
                    self.deepspeed = deepspeed_engine
                    self.optimizer = optimizer
                    self.lr_scheduler = lr_scheduler
                    self.deepspeed.load_checkpoint(
                        self.state.best_model_checkpoint, load_optimizer_states=True, load_lr_scheduler_states=True
                    )
                else:
                    # We load the model state dict on the CPU to avoid an OOM error.
                    state_dict = torch.load(best_model_path, map_location="cpu")
                    # If the model is on the GPU, it still works!
                    self._load_state_dict_in_model(state_dict)
            else:
                logger.warning(
                    f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                    "on multiple nodes, you should activate `--save_on_each_node`."
                )

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)
        
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.
        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).
        You can also subclass and override this method to inject custom behavior.

        We label all modifications with "Modified".

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is an `datasets.Dataset`, columns not
                accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)
        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()
        
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        # Evaluate on downstream datasets every k epochs after m epochs, Modified!
        if self.state.epoch % self.data_args.eval_every_k_epochs == 0 and self.state.epoch >= self.data_args.eval_after_k_epochs:
            sent_eval_metrics = {}
            if self.args.save_strategy.value != 'no':
                sent_eval_metrics = self.evaluate_senteval()
                self.model.cuda()
            output.metrics.update(sent_eval_metrics)
        else:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            # Add dummy metric if we don't compute metrics in this epoch
            dummy_metrics = {}
            dummy_metrics[metric_to_check] = self.state.best_metric if self.state.best_metric is not None else 0.0
            output.metrics.update(dummy_metrics)
            
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self.log(output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.
        Works both with or without labels.

        We label all modifications with "Modified".
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.per_device_eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0
        
        # Check how many tokens are masked in each sentence in avg, Modified!
        if not hasattr(self, "avg_token_mask_num"):
            self.avg_token_mask_ratio = None
        self.avg_token_mask_ratio = []

        # Main evaluation loop
        total_input_ids = []
        for step, inputs in enumerate(dataloader):
            # Compute the token mask num
            input_ids = inputs['input_ids']
            total_input_ids.append(input_ids)
            sent_lens = input_ids.ne(self.tokenizer.pad_token_id).sum(1, keepdims=False).cpu().numpy()
            sent_lens -= 2
            mask_lens = (input_ids == self.tokenizer.mask_token_id).sum(1, keepdims=False).cpu().numpy()
            mask_ratio = mask_lens / sent_lens
            self.avg_token_mask_ratio.append(mask_ratio)

            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                # ADD here to judge if loss is a scalar or vector, if vector, that contains the loss for each item in the batch,
                # no need to repeat, Modified!
                if len(loss.shape) > 0:
                    losses = self._nested_gather(loss)
                else:
                    losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        # Compute the avg mask ratio, MODIFIED
        self.avg_token_mask_ratio = np.concatenate(self.avg_token_mask_ratio)
        self.avg_token_mask_ratio = float(self.avg_token_mask_ratio.mean())
        logger.info(f"avg_token_mask_ratio: {self.avg_token_mask_ratio}")
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)
        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        # ADD mean and std of loss to draw the plot (mlm mask loss vs epoch, mlm predict accuracy vs loss), Modified!
        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss_mean"] = all_losses.mean().item()
            metrics[f"{metric_key_prefix}_loss_std"] = all_losses.std().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.
        Subclass and override to inject custom behavior.
        
        Our implementation will substitute the labels by the prepended labels returned by the gap model
        We label all modifications with "Modified".
        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels:
                    # Modified!
                    with self.autocast_smart_context_manager():
                        # Here the compute loss should return element-wise loss
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    # Should not take the mean here
                    loss = loss.detach()

                    if isinstance(outputs, dict):
                        remain_outputs = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        remain_outputs = outputs[1:]
                else:
                    loss = None
                    with self.autocast_smart_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        remain_outputs = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        remain_outputs = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)
        
        remain_outputs = nested_detach(remain_outputs)
        # separate logits from labels, Modified!
        if len(remain_outputs) == 1:
            logits = remain_outputs[0]
        # The last item in logits is the new labels if its length is 2, # separate logits from labels, Modified!
        elif len(remain_outputs) == 2:
            logits, labels = remain_outputs

        return (loss, logits, labels)

    def evaluate_senteval(self) -> Dict[str, float]:
        """
        Evaluate trained language model on senteval tasks.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        def prepare(params, samples):
            """
            Overwrite SentEval prepare methods
            """
            return

        # Use an enclosure to pass in the raw datasets
        def batcher(params, batch):
            """
            Overwrite SentEval batcher methods
            """
            sentences = [' '.join(s) for s in batch]
            # Obtain the sentence to id dict.
            sent_to_id = self.dataset_to_sent_id
         
            # Obtain sentence embeddings
            sent_ids = torch.tensor([sent_to_id[sent.strip()] for sent in sentences]).to(self.args.device)
            embeddings = []
            with torch.no_grad():
                embeddings = self.model.sentence_embedding.sentence_embedding(sent_ids).detach().cpu().numpy()
                

            return embeddings

        # # Define the batcher for mixing sent vectors and the original sentences as input
        # # and get the final sent embedding at CLS or avg embedding
        # def batcher_mix(params, batch):
        #     """
        #     Overwrite SentEval batcher methods
        #     """
        #     sentences = [' '.join(s) for s in batch]
        #     batch_input = self.tokenizer(sentences, return_tensors='pt', return_special_tokens_mask=False, padding=True)
            
        #     for k in batch_input:
        #         batch_input[k] = batch_input[k].to(self.args.device)

        #     with torch.no_grad():
        #         # Obtain sentence embeddings from the trained sentence vectors
        #         sent_ids = torch.tensor([self.dataset_to_sent_id[sent.strip()] for sent in sentences]).to(self.args.device)
        #         input_ids, attention_mask = batch_input['input_ids'], batch_input['attention_mask']
        #         batch_size = input_ids.shape[0]
        #         sent_embeds = self.model.sentence_embedding(sent_ids).view(batch_size, -1, self.model_args.sent_embed_size)
                
        #         # Compute plm's token embedding in advance
        #         if self.model_args.plm_name == 'bert':
        #             pretrained_model = self.model.bert
        #             token_type_ids = batch_input['token_type_ids']
        #         elif self.model_args.plm_name == 'roberta':
        #             pretrained_model = self.model.roberta
        #             # Roberta does not have token_type_ids
        #             token_type_ids = None
        #         else:
        #             raise NotImplementedError
        #         tokens_embeds = pretrained_model.embeddings.word_embeddings(input_ids)

        #         # Prepend sentence vector after the [CLS] token
        #         inputs_embeds = torch.cat([tokens_embeds[:, [0], :], sent_embeds, tokens_embeds[:, 1:, :]], dim=1)
                
        #         # Add prompt_length dummy tokens to attention_mask and  token_type_ids after [CLS] token
        #         prompt_length = self.model_args.prompt_length
        #         prompt_attention_mask = torch.ones(batch_size, prompt_length).to(attention_mask.device).long()
        #         attention_mask = torch.cat([attention_mask[:, [0]], prompt_attention_mask, attention_mask[:, 1:]], dim=1)
        #         if token_type_ids is not None:
        #             prompt_token_type_ids = torch.zeros(batch_size, prompt_length).to(token_type_ids.device).long()
        #             token_type_ids = torch.cat([token_type_ids[:, [0]], prompt_token_type_ids, token_type_ids[:, 1:]], dim=1)
        #         # # Add mask labels to the dummy token labels
        #         # prompt_labels = labels[0, 0] * torch.ones(batch_size, prompt_length).to(labels.device).long()
        #         # labels = torch.cat([labels[:, [0]], prompt_labels, labels[:, 1:]], dim=1)
                
        #         # Run plm
        #         plm_outputs = pretrained_model(
        #             input_ids=None,
        #             attention_mask=attention_mask,
        #             token_type_ids=token_type_ids,
        #             inputs_embeds=inputs_embeds,
        #             return_dict=False
        #         )

        #         # Obtain final sentence embeddings by passing through the plm
        #         if evaluate_mode == 'avg':
        #             # need to mask non-input tokens
        #             sent_lens = batch_input['input_ids'].ne(self.tokenizer.pad_token_id).sum(1, keepdims=True)
        #             # we need to include the prompt length, otherwise we miss some tokens
        #             sent_lens += self.model_args.prompt_length
        #             masks_range = torch.arange(torch.max(sent_lens))[None, :].to(self.args.device)
        #             masks = masks_range < sent_lens
        #             masked_plm_outputs = plm_outputs[0] * masks[:, :, None]
        #             plm_sent_embed = torch.sum(masked_plm_outputs, dim=1) / sent_lens
                    
        #         elif evaluate_mode == 'cls':
        #             if self.model_args.plm_name == 'bert':
        #                 plm_sent_embed = plm_outputs[1]
        #             elif self.model_args.plm_name == 'roberta':
        #                 plm_sent_embed = plm_outputs[0][:, 0, :]
        #             else:
        #                 raise NotImplementedError
        #         else:
        #             raise NotImplementedError

        #     return plm_sent_embed.cpu()

        # # Define the batcher for concatenating sent vectors with the sentence embedding
        # def batcher_concate(params, batch):
        #     sentences = [' '.join(s) for s in batch]
        #     # Obtain the sentence to id dict.
        #     sent_to_id = self.dataset_to_sent_id
        #     batch_input = self.tokenizer(sentences, return_tensors='pt', return_special_tokens_mask=False, padding=True)
            
        #     if 'special_tokens_mask' in batch_input:
        #         batch_input.pop('special_tokens_mask')
        #     for k in batch_input:
        #         batch_input[k] = batch_input[k].to(self.args.device)


        #     with torch.no_grad():
        #         # Obtain sentence embeddings from the trained sentence vectors
        #         sent_ids = torch.tensor([sent_to_id[sent.strip()] for sent in sentences]).to(self.args.device)
        #         sent_vec_outputs = self.model.sentence_embedding.sentence_embedding(sent_ids)
        #         # Obtain sentence embeddings by passing through the bert
        #         if self.model_args.plm_name == 'bert':
        #             plm_outputs = self.model.bert(**batch_input, return_dict=False)
        #         elif self.model_args.plm_name == 'roberta':
        #             plm_outputs = self.model.roberta(**batch_input, return_dict=False)
        #         elif self.model_args.plm_name == 'deberta':
        #             plm_outputs = self.model.deberta(**batch_input, return_dict=False)
        #         else:
        #             raise NotImplementedError("Please input a valid pretrained language model.")
        #         if evaluate_mode == 'avg':
        #             # need to mask non-input tokens
        #             sent_lens = batch_input['input_ids'].ne(self.tokenizer.pad_token_id).sum(1, keepdims=True)
        #             masks_range = torch.arange(torch.max(sent_lens))[None, :].to(self.args.device)
        #             masks = masks_range < sent_lens
        #             # # here we exclude the first cls token embedding
        #             # masks[:, 0] = 0
        #             # sent_lens -= 1
        #             masked_plm_outputs = plm_outputs[0] * masks[:, :, None]
        #             plm_sent_embed = torch.sum(masked_plm_outputs, dim=1) / sent_lens
        #         elif evaluate_mode == 'cls':
        #             if self.model_args.plm_name == 'bert':
        #                 plm_sent_embed = plm_outputs[1]
        #             elif self.model_args.plm_name == 'roberta':
        #                 plm_sent_embed = plm_outputs[0][:, 0, :]
        #             else:
        #                 raise NotImplementedError
        #         else:
        #             raise NotImplementedError

        #         concat_sent_embed = torch.cat((sent_vec_outputs, plm_sent_embed), dim=1)

        #     return concat_sent_embed.cpu()
            
        # Set params for SentEval
        # Fast eval
        params_senteval_fast = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params_senteval_fast['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}
        # Standard eval
        params_senteval_std = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
        params_senteval_std['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64, 'tenacity': 5, 'epoch_size': 4}
        metrics = {}

        # Pure sent embedding
        se = senteval.engine.SE(params_senteval_fast, batcher, prepare)
        tasks = self.target_tasks
        results = se.eval(tasks)

        # Post processing
        for task in self.target_tasks:
            if task in ['IMDB', 'HyperParNews']:
                # devacc
                metrics[f'eval_{task}_devacc_onlysent_fast'] = results[task]['devacc']
                # test acc
                metrics[f'eval_{task}_testacc_onlysent_fast'] = results[task]['acc']
            elif task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
                if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                    metrics[f'eval_{task}_testacc_onlysent_fast'] = results[task]['all']['spearman']['all'] * 100
                    metrics[f'eval_{task}_devacc_onlysent_fast'] = results[task]['all']['spearman']['all'] * 100
                else:
                    metrics[f'eval_{task}_testacc_onlysent_fast'] = results[task]['test']['spearman'][0] * 100
                    metrics[f'eval_{task}_devacc_onlysent_fast'] = results[task]['dev']['spearman'].correlation * 100

        self.log(metrics)
        
        self._memory_tracker.stop_and_update_metrics(metrics)

        return metrics


    def evaluate_senteval_final(self) -> Dict[str, float]:
        """
        Evaluate trained language model on senteval tasks.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        # Add final evaluate mode attribute
        if not hasattr(self, "final_eval_mode"):
            self.final_eval_mode = 'avg'

        def prepare(params, samples):
            """
            Overwrite SentEval prepare methods
            """
            return

        # Use an enclosure to pass in the raw datasets
        def batcher(params, batch):
            """
            Overwrite SentEval batcher methods
            """
            sentences = [' '.join(s) for s in batch]
            # Obtain the sentence to id dict.
            sent_to_id = self.dataset_to_sent_id
            
            # Obtain sentence embeddings
            sent_ids = torch.tensor([sent_to_id[sent.strip()] for sent in sentences]).to(self.args.device)
            embeddings = []
            with torch.no_grad():
                embeddings = self.model.sentence_embedding.sentence_embedding(sent_ids).detach().cpu().numpy()

            return embeddings

        def pack_document_plm_input(batch_input, span_length):
            """
            Pack the document level plm batch input
            """
            assert 'input_ids' in batch_input and 'attention_mask' in batch_input
            
            batch_size = batch_input['input_ids'].shape[0]
            input_ids_chunks = list(batch_input['input_ids'].split(span_length - 2, dim=1))
            attention_mask_chunks = list(batch_input['attention_mask'].split(span_length - 2, dim=1))

            cls_token = torch.tensor([self.tokenizer.cls_token_id] * batch_size).unsqueeze(1)
            sep_token = torch.tensor([self.tokenizer.sep_token_id] * batch_size).unsqueeze(1)
            attn_valid_token = torch.tensor([1] * batch_size).unsqueeze(1)
            attn_pad_token = torch.tensor([0] * batch_size).unsqueeze(1)
            input_pad_token = torch.tensor([self.tokenizer.pad_token_id] * batch_size).unsqueeze(1)
            
            # Add cls and seq tokens
            for i in range(len(input_ids_chunks)):
                # Find the sep token index
                first_sep_index = torch.sum(attention_mask_chunks[i], dim=1, keepdim=True)
                # Pad the place of sep token in case the index overflows
                input_ids_chunks[i] = torch.cat([input_ids_chunks[i], input_pad_token], dim=1)
                attention_mask_chunks[i] = torch.cat([attention_mask_chunks[i], attn_pad_token], dim=1)
                input_ids_chunks[i].scatter_(1, first_sep_index, self.tokenizer.sep_token_id)
                attention_mask_chunks[i].scatter_(1, first_sep_index, 1)
                # Concat the cls token
                input_ids_chunks[i] = torch.cat([cls_token, input_ids_chunks[i]], dim=1)
                attention_mask_chunks[i] = torch.cat([attn_valid_token, attention_mask_chunks[i]], dim=1)
                # Pad
                pad_length = span_length - input_ids_chunks[i].shape[1]
                if pad_length > 0:
                    input_ids_chunks[i] = torch.cat([input_ids_chunks[i], torch.Tensor([[self.tokenizer.pad_token_id] * pad_length for _ in range(batch_size)])], dim=1)
                    attention_mask_chunks[i] = torch.cat([attention_mask_chunks[i], torch.Tensor([[0] * pad_length for _ in range(batch_size)])], dim=1)

            # Concatenate these representations
            input_ids = torch.cat(input_ids_chunks, dim=0)
            attention_mask = torch.cat(attention_mask_chunks, dim=0)
            batch_input = {
                'input_ids': input_ids.long(),
                'attention_mask': attention_mask.int()
            }

            return batch_input

        def unpack_document_plm_embedding(plm_outputs, attention_masks, batch_size):
            """
            Unpack the plm embedding [bz * chunk_num, max_length, embed_size] to [bz, chunk_num * max_length, embed_size]
            """
            # When the attention_mask is [1, 1, 0, ...], ignore that embedding (always occur in the last chunk)
            # Put the embedding from the same document together
            plm_seq_outputs, plm_cls_outputs = plm_outputs[0], plm_outputs[1]
            attention_masks = attention_masks.split(batch_size, dim=0)
            attention_masks = torch.stack(attention_masks, dim=1)

            merged_embeddings = []
            # Average mode
            if self.final_eval_mode == 'avg':
                plm_embeddings = plm_seq_outputs.split(batch_size, dim=0)
                plm_embeddings = torch.stack(plm_embeddings, dim=1)
            elif self.final_eval_mode == 'cls':
                if self.model_args.plm_name == 'bert':
                    plm_embeddings = plm_cls_outputs
                elif self.model_args.plm_name == 'roberta':
                    plm_embeddings = plm_seq_outputs[:, 0, :]
                else:
                    raise NotImplementedError
                plm_embeddings = plm_embeddings.split(batch_size, dim=0)
                plm_embeddings = torch.stack(plm_embeddings, dim=1)
                
            for i, plm_embedding in enumerate(plm_embeddings):
                attention_mask = attention_masks[i]
                # Just use the stupid method !!!
                idx_len = plm_embedding.shape[0]
                merged_one_embedding = []
                for idx in range(idx_len):
                    # Skip when this input_ids only contain cls and sep token, ignore that embedding
                    if torch.sum(attention_mask[idx]) != 2:
                        if self.final_eval_mode == 'avg':
                            valid_embedding = plm_embedding[idx] * attention_mask[idx][:, None]
                            valid_length = attention_mask[idx].sum(-1)
                            valid_embedding = valid_embedding[:valid_length]
                            merged_one_embedding.append(valid_embedding)
                        elif self.final_eval_mode == 'cls':
                            merged_one_embedding.append(plm_embedding[[idx]])

                avg_one_embedding = torch.mean(torch.cat(merged_one_embedding, dim=0), dim=0)
                merged_embeddings.append(avg_one_embedding)

            merged_embeddings = torch.stack(merged_embeddings, dim=0)
        
            return merged_embeddings

        # Define the batcher for concatenating sent vectors with the sentence embedding
        def batcher_concate(params, batch):
            sentences = [' '.join(s) for s in batch]
            batch_size = len(sentences)
            # Obtain the sentence to id dict.
            sent_to_id = self.dataset_to_sent_id
            batch_input = self.tokenizer(sentences, return_tensors='pt', return_special_tokens_mask=False, padding=True, add_special_tokens=False)
            
            # TODO: Split them to chunks with maximum length == max_length
            packed_batch_input = pack_document_plm_input(batch_input, self.data_args.max_seq_length)
            
            if 'special_tokens_mask' in batch_input:
                packed_batch_input.pop('special_tokens_mask')
            for k in packed_batch_input:
                packed_batch_input[k] = packed_batch_input[k].to(self.args.device)

            with torch.no_grad():
                # Obtain sentence embeddings from the trained sentence vectors
                sent_ids = torch.tensor([sent_to_id[sent.strip()] for sent in sentences]).to(self.args.device)
                # sent_vec_outputs = self.model.sentence_embedding(sent_ids)
                sent_vec_outputs = self.model.sentence_embedding.sentence_embedding(sent_ids)
                # Obtain sentence embeddings by passing through the bert
                try:
                    if self.model_args.plm_name == 'bert':
                        plm_outputs = self.model.bert(**packed_batch_input, return_dict=False)
                    elif self.model_args.plm_name == 'roberta':
                        plm_outputs = self.model.roberta(**packed_batch_input, return_dict=False)
                    elif self.model_args.plm_name == 'deberta':
                        plm_outputs = self.model.deberta(**packed_batch_input, return_dict=False)
                    else:
                        raise NotImplementedError("Please input a valid pretrained language model.")
                except Exception as e:
                    print(e)
                    pdb.set_trace()
                
                # TODO: here we need to flat the output to get avg or cls embedding
                plm_document_embed = unpack_document_plm_embedding(plm_outputs, packed_batch_input['attention_mask'], batch_size)

                concat_sent_embed = torch.cat((sent_vec_outputs, plm_document_embed), dim=1)

            return concat_sent_embed.cpu()

        # # Define the batcher for mixing sent vectors and the original sentences as input
        # # and get the final sent embedding at CLS or avg embedding
        # def batcher_mix(params, batch):
        #     """
        #     Overwrite SentEval batcher methods
        #     """
        #     sentences = [' '.join(s) for s in batch]
        #     batch_input = self.tokenizer(sentences, return_tensors='pt', return_special_tokens_mask=False, padding=True)
            
        #     for k in batch_input:
        #         batch_input[k] = batch_input[k].to(self.args.device)

        #     with torch.no_grad():
        #         # Obtain sentence embeddings from the trained sentence vectors
        #         sent_ids = torch.tensor([self.dataset_to_sent_id[sent.strip()] for sent in sentences]).to(self.args.device)
        #         input_ids, attention_mask = batch_input['input_ids'], batch_input['attention_mask']
        #         batch_size = input_ids.shape[0]
        #         sent_embeds = self.model.sentence_embedding(sent_ids).view(batch_size, -1, self.model_args.sent_embed_size)
                
        #         # Compute plm's token embedding in advance
        #         if self.model_args.plm_name == 'bert':
        #             pretrained_model = self.model.bert
        #             token_type_ids = batch_input['token_type_ids']
        #         elif self.model_args.plm_name == 'roberta':
        #             pretrained_model = self.model.roberta
        #             # Roberta does not have token_type_ids
        #             token_type_ids = None
        #         else:
        #             raise NotImplementedError
        #         tokens_embeds = pretrained_model.embeddings.word_embeddings(input_ids)

        #         # Prepend sentence vector after the [CLS] token
        #         inputs_embeds = torch.cat([tokens_embeds[:, [0], :], sent_embeds, tokens_embeds[:, 1:, :]], dim=1)
                
        #         # Add prompt_length dummy tokens to attention_mask and  token_type_ids after [CLS] token
        #         prompt_length = self.model_args.prompt_length
        #         prompt_attention_mask = torch.ones(batch_size, prompt_length).to(attention_mask.device).long()
        #         attention_mask = torch.cat([attention_mask[:, [0]], prompt_attention_mask, attention_mask[:, 1:]], dim=1)
        #         if token_type_ids is not None:
        #             prompt_token_type_ids = torch.zeros(batch_size, prompt_length).to(token_type_ids.device).long()
        #             token_type_ids = torch.cat([token_type_ids[:, [0]], prompt_token_type_ids, token_type_ids[:, 1:]], dim=1)
        #         # # Add mask labels to the dummy token labels
        #         # prompt_labels = labels[0, 0] * torch.ones(batch_size, prompt_length).to(labels.device).long()
        #         # labels = torch.cat([labels[:, [0]], prompt_labels, labels[:, 1:]], dim=1)
                
        #         # Run plm
        #         plm_outputs = pretrained_model(
        #             input_ids=None,
        #             attention_mask=attention_mask,
        #             token_type_ids=token_type_ids,
        #             inputs_embeds=inputs_embeds,
        #             return_dict=False
        #         )

        #         # Obtain final sentence embeddings by passing through the plm
        #         if self.final_eval_mode == 'avg':
        #             # need to mask non-input tokens
        #             sent_lens = batch_input['input_ids'].ne(self.tokenizer.pad_token_id).sum(1, keepdims=True)
        #             # we need to include the prompt length, otherwise we miss some tokens
        #             sent_lens += self.model_args.prompt_length
        #             masks_range = torch.arange(torch.max(sent_lens))[None, :].to(self.args.device)
        #             masks = masks_range < sent_lens
        #             # # here we exclude the first cls token embedding (and the prompt vectors)
        #             # masks[:, :1+self.model_args.prompt_length] = 0
        #             # sent_lens -= (1+self.model_args.prompt_length)
        #             masked_plm_outputs = plm_outputs[0] * masks[:, :, None]
        #             plm_sent_embed = torch.sum(masked_plm_outputs, dim=1) / sent_lens
        #             # plm_sent_embed = torch.mean(plm_outputs[0][:, 1:, :], dim=1)
        #             # plm_sent_embed = plm_outputs[0][:, 1:1+self.model_args.prompt_length, :].view(plm_outputs[0].shape[0], -1)
        #         elif self.final_eval_mode == 'cls':
        #             if self.model_args.plm_name == 'bert':
        #                 plm_sent_embed = plm_outputs[1]
        #             elif self.model_args.plm_name == 'roberta':
        #                 plm_sent_embed = plm_outputs[0][:, 0, :]
        #             else:
        #                 raise NotImplementedError
        #         else:
        #             raise NotImplementedError

        #     return plm_sent_embed.cpu()
            
        # Set params for SentEval
        # Standard eval
        params_senteval_std = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10, 'batch_size': 64}
        params_senteval_std['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64, 'tenacity': 5, 'epoch_size': 4}

        # Evaluate on SentEval
        metrics = {}
        # cls with concat and mix methods
        logger.info("Evaluate concat method with cls embedding...")
        self.final_eval_mode = 'cls'
        se = senteval.engine.SE(params_senteval_std, batcher_concate, prepare)
        concat_cls_results = se.eval(self.target_tasks)
        # logger.info("Evaluate mix method with cls embedding...")
        # se = senteval.engine.SE(params_senteval_std, batcher_mix, prepare)
        # mix_cls_results = se.eval(self.target_tasks)

        # avg with concat and mix methods
        self.final_eval_mode = 'avg'
        logger.info("Evaluate concat method with avg embedding...")
        se = senteval.engine.SE(params_senteval_std, batcher_concate, prepare)
        concat_avg_results = se.eval(self.target_tasks)
        # logger.info("Evaluate mix method with avg embedding...")
        # se = senteval.engine.SE(params_senteval_std, batcher_mix, prepare)
        # mix_avg_results = se.eval(self.target_tasks)

        # original pure sent vector
        logger.info("Evaluate only sent-vectors...")
        se = senteval.engine.SE(params_senteval_std, batcher, prepare)
        sentvec_results = se.eval(self.target_tasks)
        
        # Post processing=
        for task in self.target_tasks:
            if task in ['IMDB', 'HyperParNews']:
                # devacc
                metrics[f'eval_{task}_devacc_concat_cls'] = concat_cls_results[task]['devacc']
                metrics[f'eval_{task}_devacc_concat_avg'] = concat_avg_results[task]['devacc']
                # metrics[f'eval_{task}_devacc_mix_cls'] = mix_cls_results[task]['devacc']
                # metrics[f'eval_{task}_devacc_mix_avg'] = mix_avg_results[task]['devacc']
                metrics[f'eval_{task}_devacc_sentvec'] = sentvec_results[task]['devacc']
                # test acc
                metrics[f'eval_{task}_testacc_concat_cls'] = concat_cls_results[task]['acc']
                metrics[f'eval_{task}_testacc_concat_avg'] = concat_avg_results[task]['acc']
                # metrics[f'eval_{task}_testacc_mix_cls'] = mix_cls_results[task]['acc']
                # metrics[f'eval_{task}_testacc_mix_avg'] = mix_avg_results[task]['acc']
                metrics[f'eval_{task}_testacc_sentvec'] = sentvec_results[task]['acc']
            elif task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
                if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                    metrics[f'eval_{task}_concat_cls'] = concat_cls_results[task]['all']['spearman']['all'] * 100
                    metrics[f'eval_{task}_concat_avg'] = concat_avg_results[task]['all']['spearman']['all'] * 100
                    # metrics[f'eval_{task}_mix_cls'] = mix_cls_results[task]['all']['spearman']['all'] * 100
                    # metrics[f'eval_{task}_mix_avg'] = mix_avg_results[task]['all']['spearman']['all'] * 100
                    metrics[f'eval_{task}_sentvec'] = sentvec_results[task]['all']['spearman']['all'] * 100
                else:
                    metrics[f'eval_{task}_cocat_cls'] = concat_cls_results[task]['test']['spearman'].correlation * 100
                    metrics[f'eval_{task}_concat_avg'] = concat_avg_results[task]['test']['spearman'].correlation * 100
                    # metrics[f'eval_{task}_mix_cls'] = mix_cls_results[task]['test']['spearman'].correlation * 100
                    # metrics[f'eval_{task}_mix_avg'] = mix_avg_results[task]['test']['spearman'].correlation * 100
                    metrics[f'eval_{task}_sentvec'] = sentvec_results[task]['test']['spearman'].correlation * 100

        self.log(metrics)
        
        self._memory_tracker.stop_and_update_metrics(metrics)

        return metrics

    def save_sentence_vector(self):
        """
        Save the sentence vectors to desired place
        """
        logger.info(f"Saving sentvector at epoch {self.state.epoch}...")
        sent_vector = self.model.sentence_embedding.sentence_embedding.weight.data.detach().cpu().numpy()
        file_path = os.path.join(self.args.output_dir, 'sent_vector.npy')
        
        with open(file_path, 'wb') as f:
            np.save(f, sent_vector)
