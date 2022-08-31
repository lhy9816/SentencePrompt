"""
Sentence Prompt Matrix
Author: Hangyu Li
Date: 18/04/2022
"""

import torch.nn as nn


class SentenceEmbedding(nn.Module):
    """
    Embed each sentence according to their sentence id to get sentence vector representation.
    """
    def __init__(self, config, plm_config=None):
        super().__init__()
        self.config = config
        is_sparse = True if config.optimizer_name == 'sparse_adam' else False
        self.sentence_embedding = nn.Embedding(config.corpora_size, config.prompt_length * config.sent_embed_size, sparse=is_sparse)
        if config.optimizer_name == 'sgd_layernorm':
            self.layer_norm = nn.LayerNorm(config.prompt_length * config.sent_embed_size, eps=plm_config.layer_norm_eps)

    def set_sentence_embedding(self, embeddings):
        self.sentence_embedding.weight.data.copy_(embeddings)

    def uniform_init_embedding(self, init_range):
        self.sentence_embedding.weight.data.uniform_(-init_range, init_range)

    def normal_init_embedding(self, mean, std):
        self.sentence_embedding.weight.data.normal_(mean, std)

    def forward(self, sentence_id):
        if self.config.optimizer_name == 'sgd_layernorm':
            return self.layer_norm(self.sentence_embedding(sentence_id))
        else:
            return self.sentence_embedding(sentence_id)