"""
Evaluate Bert cls token / avg embedding on SentEval tasks
Author: Hangyu Li
Date: 15/05/2022
"""
import torch
import logging
import sys
import numpy as np
import torch.nn as nn

from dataclasses import dataclass, field

from transformers import AutoConfig, AutoTokenizer, AutoModel, HfArgumentParser, MODEL_FOR_MASKED_LM_MAPPING
from typing import Optional


# Set logging information
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


# Set PATH to SentEval
PATH_TO_SENTEVAL = '../../SentEval'
PATH_TO_DATA = '../../SentEval/data'


# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


# Prepare the arguments
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
    evaluate_mode: str = field(
        default='cls',
        metadata={
            "help": "Use Bert CLS or avg word embedding as the sentence embedding."
        }
    )
    target_task: Optional[str] = field(
        default=None,
        metadata={
            "help": "The SentEval task to be selected for training / evaluation."
        },
    )
    plm_name: Optional[str] = field(
        default=None,
        metadata={"help": "The language model to be used."},
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


# credit to https://github.com/princeton-nlp/SimCSE
# Pooler class
class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last", "concat_first_last", "concat_with_zero", "concat_with_random"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, outputs, attention_mask):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states
        # last_hidden = outputs[0]
        # pooler_output = outputs[1]
        # hidden_states = outputs[2]
        
        if self.pooler_type == 'cls_before_pooler':
            return last_hidden[:, 0]
        elif self.pooler_type == 'cls':
            return pooler_output
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "concat_first_last":
            first_hidden = hidden_states[0]
            # last_hidden = hidden_states[-1]
            avg_first_hidden = (first_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            avg_last_hidden = (last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            pooled_result = torch.cat([avg_first_hidden, avg_last_hidden], dim=-1)
            return pooled_result
        elif self.pooler_type == "concat_with_zero":
            avg_last_hidden = (last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            zero_embedding = torch.zeros_like(avg_last_hidden).to(last_hidden.device)
            pooled_result = torch.cat([zero_embedding, avg_last_hidden], dim=-1)
            return pooled_result
        elif self.pooler_type == "concat_with_random":
            avg_last_hidden = (last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            random_embedding = torch.normal(0, 0.02, avg_last_hidden.shape).to(last_hidden.device)
            pooled_result = torch.cat([random_embedding, avg_last_hidden], dim=-1)
            return pooled_result
        else:
            raise NotImplementedError


# Plain Bert Model
class BertEvaluator:
    def __init__(self, model_args):
        config_kwargs = {
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "use_auth_token": True if model_args.use_auth_token else None,
        }
        tokenizer_kwargs = {
            "cache_dir": model_args.cache_dir,
            "use_fast": model_args.use_fast_tokenizer,
            "revision": model_args.model_revision,
            "use_auth_token": True if model_args.use_auth_token else None,
        }
        self.config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
        self.model = AutoModel.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=self.config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        self.pooler = Pooler(model_args.evaluate_mode)
        self.evaluate_mode = model_args.evaluate_mode
        target_tasks = model_args.target_task
        self.target_tasks = target_tasks if isinstance(target_tasks, list) else [target_tasks]
        self.plm_name = model_args.plm_name
        self.device = 'cuda:0'

    def log(self, logs):
        """
        Log metrics helper function
        """
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
        
        logger.info(f"******** Logging Info ********")
        logs_formatted = log_format(logs)
        k_width = max(len(str(x)) for x in logs_formatted.keys())
        v_width = max(len(str(x)) for x in logs_formatted.values())
        for key in sorted(logs_formatted.keys()):
            logger.info(f"  {key: <{k_width}} = {logs_formatted[key]:>{v_width}}")

    def evaluate(self):
        """
        Evaluate trained language model on senteval tasks.
        """
        # memory metrics - must set up as early as possible
        self.model.cuda().eval()

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
            batch_input = self.tokenizer(sentences, return_tensors='pt', return_special_tokens_mask=False, padding=True)
            
            if 'special_tokens_mask' in batch_input:
                batch_input.pop('special_tokens_mask')
            for k in batch_input:
                batch_input[k] = batch_input[k].to(self.device)
            
            # Obtain sentence embeddings by passing through the bert
            with torch.no_grad():
                outputs = self.model(**batch_input, output_hidden_states=True, return_dict=True)
                sent_embed = self.pooler(outputs, batch_input['attention_mask'])

            return sent_embed.cpu()
            
        # Set params for SentEval
        # Fast eval
        params_senteval_fast = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params_senteval_fast['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}

        # Standard eval
        params_senteval_std = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
        params_senteval_std['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64, 'tenacity': 5, 'epoch_size': 4}
        
        # Evaluate on SentEval-std
        se_std = senteval.engine.SE(params_senteval_std, batcher, prepare)
        tasks = self.target_tasks
        results_std = se_std.eval(tasks)
        
        # # Post processing
        metrics = {}
        for task in self.target_tasks:
            if task in ['WILA', 'WILB', 'WILC', 'MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'SST5', 'TREC', 'MRPC', 'Tense', 'SICKEntailment', 'Length', 'WordContent', 'Depth',
                    'TopConstituents','BigramShift', 'Tense', 'SubjNumber', 'ObjNumber', 'OddManOut', 
                    'CoordinationInversion']:
                metrics['eval_{}_test_std'.format(task)] = results_std[task]['acc']
                metrics['eval_{}_dev_std'.format(task)] = results_std[task]['devacc']
            elif task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
                if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                    metrics['eval_{}_test_std'.format(task)] = results_std[task]['all']['spearman']['all'] * 100
                    # metrics['eval_{}_dev_std'.format(task)] = results_std[task]['dev']['spearman'][0] * 100
                else:
                    metrics['eval_{}_dev_std'.format(task)] = results_std[task]['dev']['spearman'][0] * 100
                    metrics['eval_{}_test_std'.format(task)] = results_std[task]['test']['spearman'].correlation * 100
        
            if task in ['WILA', 'WILB', 'WILC'] and 'error_genre_acc' in results_std[task]:
                logger.info(f"{task} error_genre accuracy:")
                logger.info(results_std[task]['error_genre_acc'])
                logger.info(f"{task} error_type accuracy:")
                logger.info(results_std[task]['error_type_acc'])

        self.log(metrics)

        return metrics


def main():
    # Get input arguments
    parser = HfArgumentParser(ModelArguments)
    model_args = parser.parse_args_into_dataclasses()[0]
    # model_args.target_task = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness', 'SICKEntailment']
    # model_args.target_task = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
    #                 'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', 'Length', 'WordContent', 'Depth',
    #                 'TopConstituents','BigramShift', 'Tense', 'SubjNumber', 'ObjNumber', 'OddManOut', 
    #                 'CoordinationInversion']
    # model_args.target_task = ['CR', 'SUBJ', 'SICKEntailment', 'WordContent']
    # model_args.target_task = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC',
    #                 'Length', 'WordContent', 'Depth',
    #                 'TopConstituents','BigramShift', 'Tense', 'SubjNumber', 'ObjNumber', 'OddManOut', 
    #                 'CoordinationInversion']
    # model_args.target_task = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC',]
    # model_args.target_task = ['WILA', 'WILB', 'WILC']
    # Initialize evaluator
    evaluator = BertEvaluator(model_args)
    # Evaluate on SentEval
    evaluator.evaluate()


if __name__ == "__main__":
    main()
