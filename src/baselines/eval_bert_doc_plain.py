"""
Evaluate Bert cls token / avg embedding on SentEval tasks
Author: Hangyu Li
Date: 15/05/2022
"""
import logging
import torch
import sys

import numpy as np
import torch.nn as nn

from dataclasses import dataclass, field
from typing import Optional
from transformers import AutoConfig, AutoTokenizer, AutoModel, HfArgumentParser, MODEL_FOR_MASKED_LM_MAPPING


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
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    use_longformer: bool = field (
        default=False,
        metadata={"help": "Whether using longformer for evaluation."}
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
        else:
            raise NotImplementedError


# Plain Bert Model
class BertEvaluator:
    """
    Evaluate class for BERT and RoBERTa doc classification baselines
    """
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
        self.max_seq_length = model_args.max_seq_length
        self.use_longformer = model_args.use_longformer

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
        self.model.cuda().eval()

        def prepare(params, samples):
            """
            Overwrite SentEval prepare methods
            """
            # No need to prepare
            return

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
            plm_seq_outputs = plm_outputs.last_hidden_state
            plm_cls_outputs = plm_outputs.pooler_output
            hidden_states = plm_outputs.hidden_states
            # plm_seq_outputs, plm_cls_outputs = plm_outputs[0], plm_outputs[1]
            attention_masks = attention_masks.split(batch_size, dim=0)
            attention_masks = torch.stack(attention_masks, dim=1)

            merged_embeddings = []
            # Average mode
            if self.evaluate_mode in ['avg', 'concat_with_zero', 'concat_with_random']:
                plm_embeddings = plm_seq_outputs.split(batch_size, dim=0)
                plm_embeddings = torch.stack(plm_embeddings, dim=1)
            elif self.evaluate_mode == 'cls':
                if self.plm_name == 'bert':
                    plm_embeddings = plm_cls_outputs
                elif self.plm_name == 'roberta':
                    plm_embeddings = plm_seq_outputs[:, 0, :]
                else:
                    raise NotImplementedError
                plm_embeddings = plm_embeddings.split(batch_size, dim=0)
                plm_embeddings = torch.stack(plm_embeddings, dim=1)
            elif self.evaluate_mode == 'concat_first_last':
                first_hiddens = hidden_states[0]
                plm_embeddings = torch.cat([first_hiddens, plm_seq_outputs], dim=-1)
                plm_embeddings = plm_embeddings.split(batch_size, dim=0)
                plm_embeddings = torch.stack(plm_embeddings, dim=1)
                
            for i, plm_embedding in enumerate(plm_embeddings):
                attention_mask = attention_masks[i]
                # Need to iterate on each token embedding
                idx_len = plm_embedding.shape[0]
                merged_one_embedding = []
                for idx in range(idx_len):
                    # Skip when this input_ids only contain cls and sep token, ignore that embedding
                    if torch.sum(attention_mask[idx]) != 2:
                        if self.evaluate_mode in ['avg', 'concat_first_last', 'concat_with_zero', 'concat_with_random']:
                            valid_embedding = plm_embedding[idx] * attention_mask[idx][:, None]
                            valid_length = attention_mask[idx].sum(-1)
                            valid_embedding = valid_embedding[:valid_length]
                            merged_one_embedding.append(valid_embedding)
                        elif self.evaluate_mode == 'cls':
                            merged_one_embedding.append(plm_embedding[[idx]])
                
                avg_one_embedding = torch.mean(torch.cat(merged_one_embedding, dim=0), dim=0)
                merged_embeddings.append(avg_one_embedding)

            merged_embeddings = torch.stack(merged_embeddings, dim=0)

            if self.evaluate_mode == 'concat_with_zero':
                zero_embeddings = torch.zeros_like(merged_embeddings).to(merged_embeddings.device)
                merged_embeddings = torch.cat([zero_embeddings, merged_embeddings], dim=-1)
            elif self.evaluate_mode == 'concat_with_random':
                random_embeddings = torch.normal(0, 0.02, merged_embeddings.shape).to(merged_embeddings.device)
                merged_embeddings = torch.cat([random_embeddings, merged_embeddings], dim=-1)

            return merged_embeddings
        # Use an enclosure to pass in the raw datasets
        def batcher(params, batch):
            """
            Overwrite SentEval batcher methods
            """
            sentences = [' '.join(s) for s in batch]
            batch_size = len(sentences)
            if self.use_longformer:
                batch_input = self.tokenizer(sentences, return_tensors='pt', return_special_tokens_mask=False, padding=True, max_length=self.max_seq_length, truncation=True)
            else:
                batch_input = self.tokenizer(sentences, return_tensors='pt', return_special_tokens_mask=False, padding=True, add_special_tokens=False)
            
            if 'special_tokens_mask' in batch_input:
                batch_input.pop('special_tokens_mask')
            
            if self.use_longformer:
                for k in batch_input:
                    batch_input[k] = batch_input[k].to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**batch_input, output_hidden_states=True, return_dict=True)
                    plm_document_embed = self.pooler(outputs, batch_input['attention_mask'])
                    
            else:
                # TODO: Split them to chunks with maximum length == max_length
                packed_batch_input = pack_document_plm_input(batch_input, self.max_seq_length)
                
                for k in packed_batch_input:
                    packed_batch_input[k] = packed_batch_input[k].to(self.device)
                
                # Obtain sentence embeddings by passing through the bert
                with torch.no_grad():
                    plm_outputs = self.model(**packed_batch_input, output_hidden_states=True, return_dict=True)
                    plm_document_embed = unpack_document_plm_embedding(plm_outputs, packed_batch_input['attention_mask'], batch_size)

            return plm_document_embed.cpu()
            
        # Set params for SentEval
        # Standard eval
        params_senteval_std = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10, 'batch_size': 32}
        params_senteval_std['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64, 'tenacity': 5, 'epoch_size': 4}
        
        # Evaluate on SentEval-std
        se_std = senteval.engine.SE(params_senteval_std, batcher, prepare)
        tasks = self.target_tasks
        results_std = se_std.eval(tasks)
        
        # Post processing
        metrics = {}
        for task in self.target_tasks:
            if task in ['IMDB', 'HyperParNews',
                        'WILA', 'WILB', 'WILC', 
                        'MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC', 
                        'Length', 'WordContent', 'Depth', 'TopConstituents', 'BigramShift', 
                        'Tense', 'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion']:
                metrics['eval_{}_test_std'.format(task)] = results_std[task]['acc']
                metrics['eval_{}_dev_std'.format(task)] = results_std[task]['devacc']
            elif task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
                if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                    metrics['eval_{}_test_std'.format(task)] = results_std[task]['all']['spearman']['all'] * 100
                else:
                    metrics['eval_{}_dev_std'.format(task)] = results_std[task]['dev']['spearman'][0] * 100
                    metrics['eval_{}_test_std'.format(task)] = results_std[task]['test']['spearman'].correlation * 100

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
    # model_args.target_task = ['Length', 'WordContent', 'Depth',
    #                 'TopConstituents','BigramShift', 'Tense', 'SubjNumber', 'ObjNumber', 'OddManOut', 
    #                 'CoordinationInversion']
    # model_args.target_task = ['WILA', 'WILB', 'WILC']
    # model_args.target_task = ['HyperParNews', 'IMDB']
    # Initialize evaluator
    evaluator = BertEvaluator(model_args)
    # Evaluate on SentEval
    evaluator.evaluate()


if __name__ == "__main__":
    main()
