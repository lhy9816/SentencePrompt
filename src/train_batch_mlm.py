"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=fill-mask
"""
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.

"""
Main entrance to train sentence-level tasks in duplicate batch scheme
scripts modified based on https://github.com/huggingface/transformers/blob/v4.18-release/examples/pytorch/language-modeling/run_mlm.py
"""
from cProfile import label
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import datasets
from datasets import load_dataset, load_metric
from sklearn.metrics import log_loss

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    ProgressCallback,
    PrinterCallback,
    is_torch_tpu_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version

from torch.optim import SparseAdam, RMSprop, SGD


# Set logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# Added
from typing import Optional
from trainers.bert_batch_trainer import BertTrainer
from utils.callbacks import LoggingCallback, AvgEvalCallback, SavingMetricsCallback, SavingEachSentVecCallback, MyEarlyStoppingCallback, StopTrainingCallback, SavingModelCallback
from utils.data_collators import BatchDataCollatorForLanguageModeling, MyDataCollatorForLanguageModeling, MyDataCollatorForSpanMask, MyDataCollatorForWholeWordMask, BatchDataCollatorForSpanMask
from utils.util import get_plm_model, freeze_plm_params, add_sentences_id_to_dataset, custom_load_dataset, init_sent_vector_embedding, write_final_results_to_one_file


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.19.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    # Huggingface recommended arguments
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    # CaptureGap required arguments
    plm_name: Optional[str] = field(
        default=None,
        metadata={"help": "The language model to be used."},
    )
    lm_head_name: Optional[str] = field(
        default=None,
        metadata={"help": "The language model training head to be used."},
    )
    # corpora_size needs to add after loading the dataset
    corpora_size: Optional[int] = field(
        default=None,
        metadata={"help": "The copora size used for computing sentence embedding."}
    )
    sent_embed_size: Optional[int] = field(
        default=768,
        metadata={"help": "The gap sentence embedding size."},
    )
    prompt_length: Optional[int] = field(
        default=2,
        metadata={
            "help": "The token length of the sentence vector when it is inserted into the embedding layer of "
            "the pretrained language model"
        },
    )
    # use_reparam_trick: bool = field(
    #     default=False,
    #     metadata={
    #         "help": "Whether use reparameterization trick to map sentence embedding to a lower space and then"
    #         "back to original dim"
    #     },
    # )
    reparam_hidden_size: Optional[int] = field (
        default=None,
        metadata={"help": "The hidden size used for reparameterize the sentence embedding."},
    )
    # eval_target: Optional[str] = field (
    #     default=None,
    #     metadata={"help": "The evaluation target to be selected besides lm (both, reparam, lookup)"}
    # )
    init_with_pmean: bool = field(
        default=False,
        metadata={
            "help": "whether use the concatenate of power mean of PLM contextual word embedding to initialise"
            "the sentence prompts."
        }
    )
    init_with_zero: bool = field(
        default=False,
        metadata={
            "help": "whether to use zero word embedding to initialise the sentence prompts."
        }
    )
    use_special_lr_scheduler: Optional[str] = field(
        default=None,
        metadata={"help": "The customized lr scheduler to be used. reduceOnPlateau, exponential, polynomial"}
    )
    uniform_init_range: Optional[float] = field (
        default=None,
        metadata={
            "help": "The range of uniformed initialised sentence vectors [-uniform_init_range, uniform_init_range]"
        }
    )
    normal_init_range: Optional[float] = field (
        default=None,
        metadata={
            "help": "The std of normal initialised sentence vectors mean=0.0, std=normal_init_range"
        }
    )
    mlm_loss_no_reduction: bool = field (
        default=False,
        metadata={
            "help": "Whether use reduction='mean' to compute the mlm loss. If True, use 'mean', instead"
            ", compute the loss for each instance in the batch."
        }
    )
    optimizer_name: str = field (
        default="adamw",
        metadata={"help": "Desired optimizer configuration."}
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    mean_noise_span_length: float = field(
        default=3.0,
        metadata={"help": "Mean span length of masked tokens"},
    )

    # CaptureGap required arguments
    target_task: Optional[str] = field(
        default=None,
        metadata={
            "help": "The SentEval task to be selected for training / evaluation."
        },
    )

    eval_every_k_epochs: Optional[int] = field(
        default=None,
        metadata={
            "help": "The epoch interval to evaluate on downstream/probing tasks."
        }
    )

    eval_after_k_epochs: Optional[int] = field(
        default=None,
        metadata={
            "help": "The epoch when we should begin evaluating on downstream/probing tasks."
        }
    )
    early_stopping_patience: Optional[int] = field(
        default=None,
        metadata={"help": "The epoch to wait before applying early stopping."}
    )
    duplicate_times: Optional[int] = field(
        default=None,
        metadata={"help": "The number to duplicate one instance in the batch"}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError("`train_file` should be a csv, a json or a txt file.")
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError("`validation_file` should be a csv, a json or a txt file.")


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # BAD NEED INVESTIGATE
    training_args.load_best_model_at_end = True
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log all arguments
    logger.info(model_args)
    logger.info(data_args)
    logger.info(training_args)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Download from Huggingface Datasets library.
    raw_datasets = custom_load_dataset(data_args, model_args)

    # LHY add
    # Add idx column to raw_dataset to get the sentence's corresponding sentence embedding
    raw_datasets = add_sentences_id_to_dataset(raw_datasets)
        
    # Record the training corpora size for the sentence embedding look up table
    if model_args.corpora_size is None:
        model_args.corpora_size = len(raw_datasets["train"])

    # Load pretrained model and tokenizer
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        plm_model = get_plm_model(model_args.plm_name)          # bert->BertGapModel, roberta->RobertaGapModel
        model = plm_model.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            ignore_mismatched_sizes=False,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            model_args=model_args,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))
    
    # Freeze all parameters except for the sentence embedding
    freeze_plm_params(model, model_args.plm_name)
    logger.info(model)
    
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    if data_args.line_by_line:
        # tokenize each nonempty line and replicate it for replicate_times
        padding = "max_length" if data_args.pad_to_max_length else False

        def train_tokenize_function(examples):
            # Remove empty lines
            examples[text_column_name] = [
                line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
            ]

            # Duplicate
            sentences = []
            total_num = len(examples[text_column_name])
            dup_time = data_args.duplicate_times
            for _ in range(data_args.duplicate_times):
                sentences += examples[text_column_name]

            sent_features = tokenizer(
                sentences,
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )
            
            features = {}
            for key in sent_features:
                features[key] = [[sent_features[key][i + k*total_num] for k in range(dup_time)] for i in range(total_num)]

            return features

        def eval_tokenize_function(examples):
            # Remove empty lines
            examples[text_column_name] = [
                line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
            ]
            return tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )

        with training_args.main_process_first(desc="dataset map train tokenization"):
            train_tokenized_datasets = raw_datasets.map(
                train_tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=[text_column_name],
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running train tokenizer on dataset line_by_line",
            )
        with training_args.main_process_first(desc="dataset map eval tokenization"):
            eval_tokenized_datasets = raw_datasets.map(
                eval_tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=[text_column_name],
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running eval tokenizer on dataset line_by_line",
            )
    else:
        raise NotImplementedError("Can only support processing one sentence in one line!")
    # else:
    #     # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
    #     # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
    #     # efficient when it receives the `special_tokens_mask`.
    #     def tokenize_function(examples):
    #         return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

    #     with training_args.main_process_first(desc="dataset map tokenization"):
    #         tokenized_datasets = raw_datasets.map(
    #             tokenize_function,
    #             batched=True,
    #             num_proc=data_args.preprocessing_num_workers,
    #             remove_columns=column_names,
    #             load_from_cache_file=not data_args.overwrite_cache,
    #             desc="Running tokenizer on every text in dataset",
    #         )

    #     # Main data processing function that will concatenate all texts from our dataset and generate chunks of
    #     # max_seq_length.
    #     def group_texts(examples):
    #         # Concatenate all texts.
    #         concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    #         total_length = len(concatenated_examples[list(examples.keys())[0]])
    #         # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    #         # customize this part to your needs.
    #         if total_length >= max_seq_length:
    #             total_length = (total_length // max_seq_length) * max_seq_length
    #         # Split by chunks of max_len.
    #         result = {
    #             k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
    #             for k, t in concatenated_examples.items()
    #         }
    #         return result

    #     # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
    #     # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
    #     # might be slower to preprocess.
    #     #
    #     # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    #     # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    #     with training_args.main_process_first(desc="grouping texts together"):
    #         tokenized_datasets = tokenized_datasets.map(
    #             group_texts,
    #             batched=True,
    #             num_proc=data_args.preprocessing_num_workers,
    #             load_from_cache_file=not data_args.overwrite_cache,
    #             desc=f"Grouping texts in chunks of {max_seq_length}",
    #         )
    
    if training_args.do_train:
        if "train" not in train_tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = train_tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        eval_dataset = eval_tokenized_datasets["train"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        # metric = load_metric("accuracy", cache_dir=model_args.cache_dir)

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics
            # ADDED We need to compute the accuracy for each sentence here, so no reshape method should be used
            mask = labels != -100
            # Compute accracy for each sentence
            metric_result = []
            
            for i in range(preds.shape[0]):
                pred, label = preds[i][mask[i]], labels[i][mask[i]]
                denom = pred.shape[0] if pred.shape[0] > 0 else 1
                metric_result.append(sum(pred==label) / denom)
        
            # Compute mean and acc of each sentence
            metric_result = np.asarray(metric_result)
            metrics = {}
            metrics['accuracy_mean'] = np.mean(metric_result, axis=-1)
            # For logging, set dummy 'eval_accuracy'
            metrics['accuracy'] = metrics['accuracy_mean']
            metrics['accuracy_std'] = np.std(metric_result, axis=-1)
            return metrics

    # Data collator
    # This one will take care of randomly masking the tokens.
    pad_to_multiple_of_8 = data_args.line_by_line and training_args.fp16 and not data_args.pad_to_max_length
    if model_args.lm_head_name == 'random_mask':
        data_collator = BatchDataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
            pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
        )
        eval_data_collator = MyDataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
            pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
        )
    elif model_args.lm_head_name == 'ww_mask':
        data_collator = MyDataCollatorForWholeWordMask(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
            pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
            plm_name=model_args.plm_name
        )
    else:
        data_collator = BatchDataCollatorForSpanMask(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
            pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
            plm_name=model_args.plm_name
        )
        eval_data_collator = MyDataCollatorForSpanMask(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
            pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
            plm_name=model_args.plm_name
        )
    
    # Initialise the sentence embedding
    init_sent_vector_embedding(model, model_args, tokenizer, training_args, eval_dataset)
    
    # Choose the optimizer
    if model_args.optimizer_name == 'sparse_adam':
        optimizer = SparseAdam(params=model.sentence_embedding.parameters(), lr=training_args.learning_rate)
    elif model_args.optimizer_name == 'sgd' or model_args.optimizer_name == 'sgd_layernorm':
        optimizer = SGD(params=model.sentence_embedding.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
    elif model_args.optimizer_name == 'sgd_momentum':
        optimizer = SGD(params=model.sentence_embedding.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay, momentum=0.9)
    elif model_args.optimizer_name == 'sgd_momentum_nesterov':
        optimizer = SGD(params=model.sentence_embedding.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay, momentum=0.5, nesterov=True)
    elif model_args.optimizer_name == 'rmsprop':
        optimizer = RMSprop(params=model.sentence_embedding.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
    else:
        optimizer = AdamW(params=model.sentence_embedding.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
    
    # Define callback functions
    saving_metrics_callback = SavingMetricsCallback(model_path=training_args.output_dir, save_prefix='eval')
    saving_all_sentvec_callback = SavingEachSentVecCallback(model_path=training_args.output_dir, model=model)
    my_early_stopping_callback = MyEarlyStoppingCallback(early_stopping_patience=data_args.early_stopping_patience, early_stopping_threshold=0.0, check_interval=data_args.eval_every_k_epochs, min_check_epoch=data_args.eval_after_k_epochs)
    # Set stop training argument to 30 to reproduce the results in our report
    stop_training_callback = StopTrainingCallback(30)

    # Initialise our trainer
    trainer = BertTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[LoggingCallback, my_early_stopping_callback, saving_metrics_callback],
        optimizers=[optimizer, None],
        raw_dataset=raw_datasets,
        model_args=model_args,
        data_args=data_args,
        eval_data_collator=eval_data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
    )
    # Remove default print callback
    # [PrinterCallback, ProgressCallback]
    trainer.remove_callback(PrinterCallback if training_args.disable_tqdm else ProgressCallback)
    
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        # Save and plot the evaluation metrics during training
        saving_metrics_callback.save_metrics()

    # Evaluation
    if training_args.do_eval and training_args.save_strategy.value != 'no':
        logger.info("*** Evaluate ***")
        metrics = {}
        # Separately save the sentence vector
        trainer.save_sentence_vector()
        # Evaluate SentEval tasks in the end
        trainer.model.eval()
        sent_eval_metrics = trainer.evaluate_senteval_final()

        metrics.update(sent_eval_metrics)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Save final result to file
    write_final_results_to_one_file(model_args, data_args, training_args, sent_eval_metrics, save_file_path='../final_batch_result.csv')
    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "fill-mask"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()