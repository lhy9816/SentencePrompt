"""
Provide useful callbacks for the training and evaluation.
Author: Hangyu Li
Date: 06-05-2022
"""
import collections
import copy
import logging
import os
import pickle

import numpy as np

from transformers import TrainerCallback, EarlyStoppingCallback

from utils.plot_figures import draw_plot, draw_plot_mean_std_fill_between, draw_plot_mean_std_subplot

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Customized Callbacks
class LoggingCallback(TrainerCallback):
    """
    A TrainerCallback that prints the logs at INFO level
    """
    def on_log(self, args, state, control, logs=None, **kwargs):
        def log_format(logs):
            """
            Reformat Trainer logs to human readable format
            """
            format_logs = logs.copy()
            
            for k, v in format_logs.items():
                if 'loss' in k:
                    format_logs[k] = round(v, 4)
                elif 'epoch' in k:
                    format_logs[k] = round(v, 2)
                elif 'eval' in k:
                    format_logs[k] = round(v, 3)
                elif '_rate' in k:
                    format_logs[k] = float(format(v, '.4g'))

            return format_logs

        _ = logs.pop("total_flos", None)
        
        if state.is_local_process_zero:
            logger.info(f"******** Logging Info ********")
            logs_formatted = log_format(logs)
            k_width = max(len(str(x)) for x in logs_formatted.keys())
            v_width = max(len(str(x)) for x in logs_formatted.values())
            for key in sorted(logs_formatted.keys()):
                logger.info(f"  {key: <{k_width}} = {logs_formatted[key]:>{v_width}}")


class AvgEvalCallback(TrainerCallback):
    """
    A trainer callback that can track the running average of evaluation accuracy.
    """
    def __init__(self, do_running_mean=False):
        self.count = 0
        self.running_mean = 0.0
        self.do_running_mean = do_running_mean

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # Get current evaluation metric value
        metric_to_check = args.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics.get(metric_to_check)

        # Compute the running mean and assign back to metrics
        if self.do_running_mean:
            self.count += 1
            self.running_mean = metric_value / self.count + self.running_mean * (self.count - 1) / self.count
            metrics[metric_to_check] = self.running_mean
        else:
            metrics[metric_to_check] = metric_value


class ReleaseReparamLayerCallback(TrainerCallback):
    """
    A trainer callback that release the parameter of reparam layer after some epoch
    """
    def __init__(self, epoch_to_release=5):
        self.epoch_to_release = epoch_to_release

    def on_epoch_begin(self, args, state, control, **kwargs):
        model = kwargs['model']
        if state.epoch >= self.epoch_to_release:
            if state.epoch == self.epoch_to_release:
                logger.info(f"Release reparam layer at epoch {state.epoch}.")
            for param in model.sentence_embedding.sentence_embedding[1].parameters():
                param.requires_grad = True


class SavingMetricsCallback(TrainerCallback):
    """
    A trainer callback that save the evaluation metrics for each epoch
    """
    def __init__(self, model_path, save_prefix='eval'):
        self.save_prefix = save_prefix
        self.metrics_to_save = collections.defaultdict(list)
        self.save_file = os.path.join(model_path, "eval_result_per_epoch.bin")
        self.output_dir = model_path

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        for metric in metrics:
            if metric.startswith(self.save_prefix):
                self.metrics_to_save[metric].append(float(metrics[metric]))

    def save_metrics(self):
        # Plot and save all related figures
        for metric_name in ["loss", "accuracy"]:
            draw_plot(None, self.metrics_to_save, 
                f"eval_{metric_name}_mean", f"mlm eval {metric_name} mean vs epoch",
                os.path.join(self.output_dir, f"eval_{metric_name}_mean_epoch.png")
            )
            draw_plot(None, self.metrics_to_save, 
                f"eval_{metric_name}_std", f"mlm eval {metric_name} std vs epoch",
                os.path.join(self.output_dir, f"eval_{metric_name}_std_epoch.png")
            )
            draw_plot_mean_std_fill_between(None, self.metrics_to_save, 
                f"eval_{metric_name}_mean", f"eval_{metric_name}_std", f"mlm eval {metric_name} mean+std vs epoch", 
                os.path.join(self.output_dir, f"eval_{metric_name}_mean_std_per_epoch.png")
            )
        draw_plot_mean_std_subplot(None, self.metrics_to_save, 
            ["eval_loss_mean", "eval_accuracy_mean"], ["eval_loss_std", "eval_accuracy_std"], "mlm eval loss and accuracy vs epoch", 
            os.path.join(self.output_dir, "eval_loss_acc_mean_std_per_epoch_subplot.png")
        )
        with open(self.save_file, 'wb') as f:
            pickle.dump(self.metrics_to_save, f)


class SavingEachSentVecCallback(TrainerCallback):
    """
    A trainer callback that save the sentence vector for each epoch
    """
    def __init__(self, model_path, model):
        self.save_sentvec_file = os.path.join(model_path, "all_sentvec.npy")
        self.save_fig_file = os.path.join(model_path, "sentvec_dimreduction_2.png")
        self.save_component_file = os.path.join(model_path, "first_two_components.npy")
        # save the initialised sentence embeddings
        sent_vec = copy.deepcopy(model.sentence_embedding.sentence_embedding.weight).detach().cpu().numpy()
        self.sentvec_to_save = sent_vec

    def on_evaluate(self, args, state, control, metrics, model, **kwargs):
        sent_vec = copy.deepcopy(model.sentence_embedding.sentence_embedding.weight).detach().cpu().numpy()
        # here we don't save the average sentence prompts, instead we save them to disk and load them
        # at each epoch 
        if os.path.isfile(self.save_sentvec_file):
            with open(self.save_sentvec_file, 'rb') as f:
                loaded_sentvecs = np.load(f)
            new_sentvecs = np.concatenate([loaded_sentvecs, sent_vec], axis=0)
        else:
            new_sentvecs = sent_vec
        
        with open(self.save_sentvec_file, 'wb') as f:
            np.save(f, new_sentvecs)


class MyEarlyStoppingCallback(EarlyStoppingCallback):
    """
    Modify the epoch interval to check the earlystopping metric
    """
    def __init__(self, early_stopping_patience, early_stopping_threshold, check_interval=10, min_check_epoch=100):
        super(MyEarlyStoppingCallback, self).__init__(early_stopping_patience, early_stopping_threshold)
        self.check_interval = check_interval
        self.min_check_epoch = min_check_epoch

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_to_check = args.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics.get(metric_to_check)
        
        if metric_value is None:
            logger.warning(
                f"early stopping required metric_for_best_model, but did not find {metric_to_check} so early stopping is disabled"
            )
            return
            
        if state.epoch % self.check_interval != 0:
            logger.info(
                f"current epoch {state.epoch} is not multiple of check interval {self.check_interval}, so we skip it now."
            )
            return

        if state.epoch < self.min_check_epoch:
            logger.info(
                f"current epoch {state.epoch} has not reached the epoch ({self.min_check_epoch})) to evaluate tasks, so we skip it now."
            )
            return

        self.check_metric_value(args, state, control, metric_value)
        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            control.should_training_stop = True


class StopTrainingCallback(TrainerCallback):
    """
    A trainer callback that save the sentence vector for each epoch
    """
    def __init__(self, num_epoch):
        self.num_epoch = num_epoch

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # Force to stop training after training for 30 epochs
        if state.epoch >= self.num_epoch:
            control.should_training_stop = True


class SavingModelCallback(TrainerCallback):
    """
    A trainer callback that saves the sentence vector every k epoch
    """
    def __init__(self, save_every_k_epoch):
        self.save_every_k_epoch = save_every_k_epoch
        
    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch % self.save_every_k_epoch == 0 and state.epoch > 10:
            control.should_save = True
        else:
            control.should_save = False