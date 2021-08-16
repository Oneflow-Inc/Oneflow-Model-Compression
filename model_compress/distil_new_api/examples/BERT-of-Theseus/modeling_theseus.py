import oneflow.experimental as flow
import argparse
import numpy as np
import os
import time
import sys
import oneflow.nn as nn
import json
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "model_compress/distil_new_api/src")))
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "./src")))
from bert_model.transformer import TransformerBlock
from bert_model.embedding.bert import BERTEmbedding


class TheseusEncoder(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1, student_n_layers=3):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()

        self.replacing_rate = None

        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        self.student_n_layers = student_n_layers
        assert self.n_layers % self.student_n_layers == 0
        self.compress_ratio = self.n_layers // self.student_n_layers

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(hidden, attn_heads, hidden * 4, dropout)
                for _ in range(self.n_layers)
            ]
        )
        self.student_transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(hidden, attn_heads, hidden * 4, dropout)
                for _ in range(self.student_n_layers)
            ]
        )

    def set_replacing_rate(self, replacing_rate):
        if not 0 < replacing_rate <= 1:
            raise Exception('Replace rate must be in the range (0, 1]!')
        self.replacing_rate = flow.tensor([replacing_rate])

    def forward(self, x, segment_info, training=True):

        if training:
            inference_layers = []
            for i in range(self.student_n_layers):
                if flow.bernoulli(self.replacing_rate) > 0.5:  # REPLACE
                    inference_layers.append(self.student_transformer_blocks[i])
                else:  # KEEP the original
                    for offset in range(self.compress_ratio):
                        inference_layers.append(self.transformer_blocks[i * self.compress_ratio + offset])
        else:  # Inference with compressed model
            inference_layers = self.student_transformer_blocks

        mask = (
            (x > 0)
                .unsqueeze(1)
                .repeat((1, x.shape[1], 1))
                .unsqueeze(1)
                .repeat((1, self.attn_heads, 1, 1))
        )

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info)
        sequence_output = [x]
        attn_list = []
        for i, layer_module in enumerate(inference_layers):
            x, attn = layer_module.forward(x, mask)
            sequence_output.append(x)
            attn_list.append(attn)

        return x, sequence_output, attn_list


class TheseusForClassification(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1, student_n_layers=3):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()

        self.encoder = TheseusEncoder(
            vocab_size=vocab_size, hidden=hidden, n_layers=n_layers, attn_heads=attn_heads, dropout=dropout,
            student_n_layers=student_n_layers
        )

        self.output_layer = nn.Linear(hidden, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, segment_info, training=True):

        encoder_outputs = self.encoder.forward(x, segment_info, training=training)

        outputs = self.output_layer(encoder_outputs[0][:, 0])
        logits = self.softmax(outputs)

        return logits, encoder_outputs
