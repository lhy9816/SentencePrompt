"""
Data collators used for different pretraining objectives
Author: Hangyu Li
Date: 29/05/2022
"""
import copy
import math
import random
import torch

import numpy as np

from dataclasses import dataclass
from transformers import (
    BatchEncoding,
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
)
from typing import Any, Dict, List, Optional, Tuple, Union


def tolist(x):
    if isinstance(x, list):
        return x
    elif hasattr(x, "numpy"):  # Checks for TF tensors without needing the import
        x = x.numpy()
    return x.tolist()


def _torch_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.
    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result


@dataclass
class MyDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    """
    Customized data collator used for language modeling.
    The only modification here is to replace masked tokens by [MASK] with 100%.
    (We only override the function torch_mask_tokens() used by torch_call())
    """

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }
        
        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # MODIFICATION HERE
        # 100% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        return inputs, labels


@dataclass
class MyDataCollatorForWholeWordMask(DataCollatorForWholeWordMask):
    """
    Customized data collator used for language modeling.
    The first modification here is to replace masked tokens by [MASK] with 100%.
    (We only override the function torch_mask_tokens() used by torch_call())
    The second modification is that we change the return format of torch_call to make it resemble
    the class DataCollatorForLanguageModeling()
    """
    # New attributes
    plm_name: str = 'bert'

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }
        
        # If special token mask has been preprocessed, pop it from the dict.
        batch.pop("special_tokens_mask", None)
     
        mask_labels = []
        for e in examples:
            ref_tokens = []
            for id in tolist(e["input_ids"]):
                token = self.tokenizer._convert_id_to_token(id)
                ref_tokens.append(token)

            # For Chinese tokens, we need extra inf to mark sub-word, e.g [喜,欢]-> [喜，##欢]
            if "chinese_ref" in e:
                ref_pos = tolist(e["chinese_ref"])
                len_seq = len(e["input_ids"])
                for i in range(len_seq):
                    if i in ref_pos:
                        ref_tokens[i] = "##" + ref_tokens[i]
            mask_labels.append(self._whole_word_mask(ref_tokens))
        batch_mask = _torch_collate_batch(mask_labels, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        batch["input_ids"], batch["labels"] = self.torch_mask_tokens(batch["input_ids"], batch_mask)
        return batch

    def torch_mask_tokens(self, inputs: Any, mask_labels: Any) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """
        
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone()
    
        probability_matrix = mask_labels

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # MODIFICATION HERE
        # 100% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        return inputs, labels


    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """
        cand_indexes = []
        for i, token in enumerate(input_tokens):
            if token == self.tokenizer.cls_token or token == self.tokenizer.sep_token:
                continue
            if self.plm_name == 'bert':
                if len(cand_indexes) >= 1 and token.startswith("##"):
                    cand_indexes[-1].append(i)
                else:
                    cand_indexes.append([i])
            elif self.plm_name == 'roberta':
                if len(cand_indexes) == 0 or token.startswith('Ġ'):
                    cand_indexes.append([i])
                else:
                    cand_indexes[-1].append(i)

        random.shuffle(cand_indexes)
        num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * self.mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        if len(covered_indexes) != len(masked_lms):
            raise ValueError("Length of covered_indexes is not equal to length of masked_lms.")
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels


@dataclass
class MyDataCollatorForSpanMask(DataCollatorForWholeWordMask):
    """
    Customized data collator used for language modeling.
    The only modification here is to replace masked tokens by [MASK] with 100%.
    We also refer to https://github.com/facebookresearch/SpanBERT/blob/main/pretraining/fairseq/data/masking.py
    to implement the span mask.
    """
    # New attributes
    plm_name: str = 'bert'

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }
        
        # If special token mask has been preprocessed, pop it from the dict.
        batch.pop("special_tokens_mask", None)
        
        # Iterate each input_ids, compute the span masks boundaries
        mask_labels = []
        batch_input_ids = batch["input_ids"]
        batch_input_ids = batch_input_ids.view(-1, batch_input_ids.shape[-1])
        
        for input_ids in batch_input_ids:
            sent_length = int(sum(input_ids != self.tokenizer.pad_token_id))
            # We need to exclude the [CLS] and [SEP] tokens
            pure_input_ids = input_ids[1: sent_length - 1].tolist()
            pure_mask_labels = self._span_mask(pure_input_ids)
            tmp_mask_labels = [0] + pure_mask_labels + [0]
            mask_labels.append(tmp_mask_labels)

        batch_mask = _torch_collate_batch(mask_labels, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        batch["input_ids"], batch["labels"] = self.torch_mask_tokens(batch["input_ids"], batch_mask)
        return batch

    
    def get_word_piece_map(self, sentence):
        """
        Return a list of bool value denoting whether the tokens at corresponding
        positions are start of words
        """
        if not hasattr(self, 'token_is_start_dict'):
            vocab_items = self.tokenizer.vocab.items()
            self.token_is_start_dict = {}
            for string, index in vocab_items:
                if self.plm_name == 'bert':
                    self.token_is_start_dict[index] = False if string.startswith('##') else True
                elif self.plm_name == 'roberta':
                    self.token_is_start_dict[index] = True if string.startswith('Ġ') else False

        ret_word_piece_map = [self.token_is_start_dict[i] for i in sentence]
        # The first token in the sentence must be the start of a word
        ret_word_piece_map[0] = True
                
        return ret_word_piece_map


    def get_word_start(self, sentence, anchor, word_piece_map):
        left = anchor
        while left > 0 and not word_piece_map[left]:
            left -= 1
        return left


    def get_word_end(self, sentence, anchor, word_piece_map):
        right = anchor + 1
        while right < len(sentence) and not word_piece_map[right]:
            right += 1
        return right


    def _span_mask(self, input_ids: Any) -> Any:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """

        sent_length = len(input_ids)
        mask_num = math.ceil(sent_length * self.mlm_probability)
        mask = set()

        # Preset span length and length distribution, from https://arxiv.org/pdf/1907.10529.pdf
        lens = list(range(1, 11))
        len_distrib = [0.22405804992033423, 0.1792464399362674, 0.14339715194901392, 0.11471772155921116, 0.09177417724736892, 0.07341934179789514, 0.05873547343831612, 0.04698837875065289, 0.037590703000522314, 0.030072562400417856]
        # change word_piece_map so that it contains the boundary information of each token
        word_piece_map = self.get_word_piece_map(input_ids)
        spans = []
        while len(mask) < mask_num:
            span_len = np.random.choice(lens, p=len_distrib)
            tagged_indices = None
            anchor  = np.random.choice(sent_length)
            if anchor in mask:
                continue
            # find word start, end, NEED MODIFICATION
            # need to understand how the left and right is defined
            # True represents is_start
            left1, right1 = self.get_word_start(input_ids, anchor, word_piece_map), self.get_word_end(input_ids, anchor, word_piece_map)
            spans.append([left1, left1])
            for i in range(left1, right1):
                if len(mask) >= mask_num:
                    break
                mask.add(i)
                spans[-1][-1] = i
            num_words = 1
            right2 = right1
            while num_words < span_len and right2 < len(input_ids) and len(mask) < mask_num:
                # complete current word
                left2 = right2
                # NEED MODIFICATION
                right2 = self.get_word_end(input_ids, right2, word_piece_map)
                num_words += 1
                for i in range(left2, right2):
                    if len(mask) >= mask_num:
                        break
                    mask.add(i)
                    spans[-1][-1] = i
        mask_labels = [1 if i in mask else 0 for i in range(sent_length)]
        # Here I want to rewrite the span_masking function
        return mask_labels


    def torch_mask_tokens(self, inputs: Any, mask_labels: Any) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """
        
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

        probability_matrix = mask_labels

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # MODIFICATION HERE
        # 100% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        return inputs, labels


@dataclass
class BatchDataCollatorForSpanMask(MyDataCollatorForSpanMask):
    """
    Batch the same sentence in the data collator
    """

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Flatten features
        special_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'mlm_input_ids', 'mlm_labels', 'special_tokens_mask']
        bs = len(examples)
        if bs > 0:
            num_sent = len(examples[0]['input_ids'])
        else:
            return
        flat_features = []
        
        for feature in examples:
            for i in range(num_sent):
                flat_features.append({k: feature[k][i] if k in special_keys else feature[k] for k in feature})
        
        batch = self.tokenizer.pad(
            flat_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        # If special token mask has been preprocessed, pop it from the dict.
        batch.pop("special_tokens_mask", None)
        
        # Iterate each input_ids, compute the span masks boundaries
        mask_labels = []
        batch_input_ids = batch["input_ids"]
        
        for input_ids in batch_input_ids:
            sent_length = int(sum(input_ids != self.tokenizer.pad_token_id))
            # We need to exclude the [CLS] and [SEP] tokens
            pure_input_ids = input_ids[1: sent_length - 1].tolist()
            pure_mask_labels = self._span_mask(pure_input_ids)
            tmp_mask_labels = [0] + pure_mask_labels + [0]
            mask_labels.append(tmp_mask_labels)

        batch_mask = _torch_collate_batch(mask_labels, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        batch["input_ids"], batch["labels"] = self.torch_mask_tokens(batch["input_ids"], batch_mask)
        return batch

@dataclass
class BatchDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    """
    Customized data collator used for language modeling.
    The only modification here is to replace masked tokens by [MASK] with 100%.
    (We only override the function torch_mask_tokens() used by torch_call())
    """

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Flatten features
        special_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'mlm_input_ids', 'mlm_labels', 'special_tokens_mask']
        bs = len(examples)
        if bs > 0:
            num_sent = len(examples[0]['input_ids'])
        else:
            return
        flat_features = []
        
        for feature in examples:
            for i in range(num_sent):
                flat_features.append({k: feature[k][i] if k in special_keys else feature[k] for k in feature})
        
        batch = self.tokenizer.pad(
            flat_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # MODIFICATION HERE
        # 100% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        return inputs, labels


@dataclass
class DocumentDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    """
    Customized data collator used for document-level language modeling.
    It will first select tokens in a fix-sized window moving randomly on the document, then pack them into one batch and perform torch_mask_tokens
    """
    # New attributes
    duplicate_times: int = 64
    span_length: int = 256

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        filtered_examples = self.generate_random_span_examples(examples)

        batch = self.tokenizer.pad(
            filtered_examples,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # flat batch
        for k in batch:
            if k == 'sentences_ids':
                batch[k] = batch[k].view(-1)
            else:
                batch[k] = batch[k].view(-1, batch[k].shape[-1])
        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        
        assert batch['input_ids'].shape == batch['attention_mask'].shape
        if 'token_type_ids' in batch:
            assert batch['input_ids'].shape == batch['token_type_ids'].shape

        batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
            batch["input_ids"], special_tokens_mask=special_tokens_mask
        )

        return batch

    def generate_random_span_examples(self, examples: Any):
        """
        Generate duplicate_times random consecutive span with the length of span_length
        """

        def select_random_span(input_ids):
            # remove special tokens (cls and sep)
            assert type(input_ids) == list
            input_ids_copy = copy.deepcopy(input_ids)
            input_ids_copy.remove(self.tokenizer.cls_token_id)
            input_ids_copy.remove(self.tokenizer.sep_token_id)

            # find the start range of the span
            start_idx_end = max(len(input_ids_copy) - true_span_length - 2 + 1, 1)        # -2: ignore cls and sep
            
            # create spans
            if self.duplicate_times <= start_idx_end:
                sel_start_idx = np.random.choice(start_idx_end, self.duplicate_times, replace=False)
            else:
                sel_start_idx = np.random.choice(start_idx_end, self.duplicate_times, replace=True)
            # add the cls token back
            sel_spans = [(i + 1, i + true_span_length - 2 + 1) for i in sel_start_idx]

            return sel_spans

        new_examples = []
        # special_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'special_tokens_mask']
        for example in examples:
            true_span_length = min(self.span_length, len(example['input_ids']))
            new_example = {}
            for key in example:
                if key == 'sentences_ids':
                    new_example[key] = [example[key] for _ in range(self.duplicate_times)]
                elif key == 'input_ids':
                    sel_spans = select_random_span(example[key])
                    new_example[key] = [[self.tokenizer.cls_token_id] + example[key][start:end] + [self.tokenizer.sep_token_id] for start, end in sel_spans]
                elif key == 'attention_mask':
                    new_example[key] = [[1] * true_span_length for _ in range(self.duplicate_times)]
                elif key == 'token_type_ids':
                    new_example[key] = [[0] * true_span_length for _ in range(self.duplicate_times)]
                elif key == 'special_tokens_mask':
                    new_example[key] = [[1] + [0] * (true_span_length - 2) + [1] for _ in range(self.duplicate_times)]
            for i in range(self.duplicate_times):
                tmp_new_example = {}
                for key in new_example:
                    tmp_new_example[key] = new_example[key][i]
                new_examples.append(tmp_new_example)

        return new_examples

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # MODIFICATION HERE
        # 100% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        return inputs, labels