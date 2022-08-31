"""
The model to integrate sentence prompts with RoBERTa
Author: Hangyu Li
Date: 23/04/2022
"""

import torch

from torch.nn import CrossEntropyLoss
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead

from models.sentence_embedding import SentenceEmbedding


def get_lm_head(lm_head_name, config):
    """
    Get language model head function in ['mask', 'span_mask', 'crpt_detect']
    """
    if 'mask' in lm_head_name:
        return RobertaLMHead(config)
    elif lm_head_name == 'span_mask':
        pass
    elif lm_head_name == 'crpt_detect':
        pass
    else:
        raise NotImplementedError("Please select lm_head_name in ['mask', 'span_mask', 'crpt_detect'].")


class RobertaGapModel(RobertaPreTrainedModel):

    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.bias", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, *model_args, **model_kwargs):
        super().__init__(config)
        self.model_args = model_kwargs['model_args']
        self.huggingface_config = config
        self.sentence_embedding = SentenceEmbedding(self.model_args, config)
        self.roberta = RobertaModel(config, add_pooling_layer=True)
        self.lm_head = get_lm_head(self.model_args.lm_head_name, config)

        self.post_init()

    def forward(self,
        sentences_ids,
        input_ids=None,
        attention_mask=None,
        special_tokens_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        # Input_ids: [bz, seq_len]
        batch_size = input_ids.shape[0]

        # Get trainable sentence embedding and convert to roberta embedding like size
        sent_embeds = self.sentence_embedding(sentences_ids).view(batch_size, -1, self.huggingface_config.hidden_size)

        # Compute roberta's token embedding in advance
        tokens_embeds = self.roberta.embeddings.word_embeddings(input_ids)

        # Prepend sentence vector after the [CLS] token
        inputs_embeds = torch.cat([tokens_embeds[:, [0], :], sent_embeds, tokens_embeds[:, 1:, :]], dim=1)
        
        # Add prompt_length dummy tokens to attention_mask and  token_type_ids after [CLS] token
        prompt_length = self.model_args.prompt_length
        prompt_attention_mask = torch.ones(batch_size, prompt_length).to(attention_mask.device).long()
        attention_mask = torch.cat([attention_mask[:, [0]], prompt_attention_mask, attention_mask[:, 1:]], dim=1)
        if token_type_ids is not None:
            prompt_token_type_ids = torch.zeros(batch_size, prompt_length).to(token_type_ids.device).long()
            token_type_ids = torch.cat([token_type_ids[:, [0]], prompt_token_type_ids, token_type_ids[:, 1:]], dim=1)
        # Add mask labels to the dummy token labels
        prompt_labels = labels[0, 0] * torch.ones(batch_size, prompt_length).to(labels.device).long()
        labels = torch.cat([labels[:, [0]], prompt_labels, labels[:, 1:]], dim=1)

        # Run roberta
        mha_outputs = self.roberta(
            input_ids=None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False
        )

        # Run lm_head function
        sequence_output = mha_outputs[0]
        lm_head_scores = self.lm_head(sequence_output)

        # Compute lm_head loss
        masked_lm_loss = None
        if labels is not None:
            if self.model_args.mlm_loss_no_reduction and not self.training:
                loss_fct = CrossEntropyLoss(reduction='none')  # -100 index = padding token
                masked_lm_loss = loss_fct(lm_head_scores.view(-1, self.config.vocab_size), labels.view(-1)).view(lm_head_scores.shape[:2])
                sent_len = masked_lm_loss.ne(0.0).sum(dim=1)
                masked_lm_loss = masked_lm_loss.sum(dim=1) / sent_len
                # convert nan value to 0 since there is no mask in that sentence
                masked_lm_loss = torch.nan_to_num(masked_lm_loss)
            else:
                loss_fct = CrossEntropyLoss()  # -100 index = padding token
                masked_lm_loss = loss_fct(lm_head_scores.view(-1, self.config.vocab_size), labels.view(-1))
        # import copy
        # self.last_sent_embed = copy.deepcopy(self.sentence_embedding.sentence_embedding.weight.data)
        
        if not return_dict:
            output = (lm_head_scores,) + mha_outputs[2:] + (labels,)
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        else:
            raise ValueError("We need to return the prepended labels so we cannot use MaskedLMOutput (set return_dict to False)")