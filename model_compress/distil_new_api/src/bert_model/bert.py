from numpy.lib.function_base import append
import oneflow.experimental.nn as nn
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(curPath)))
from transformer import TransformerBlock
from embedding.bert import BERTEmbedding
import numpy as np
import oneflow.experimental as flow
'''
--seq_length=128 \
  --student_num_hidden_layers=4 \
  --student_num_attention_heads=12 \
  --student_vocab_size=30522 \
  --student_hidden_size=312 \

  --teacher_num_hidden_layers=12 \
  --teacher_num_attention_heads=12 \
  --teacher_max_position_embeddings=512 \
  --teacher_type_vocab_size=2 \ 
  --teacher_vocab_size=30522 \
  --teacher_attention_probs_dropout_prob=0.1 \
  --teacher_hidden_dropout_prob=0.1 \
  --teacher_hidden_size_per_head=64 \
  --teacher_hidden_size=768 \
      '''
class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(hidden, attn_heads, hidden * 4, dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, segment_info):  # x.shape >> flow.Size([16, 20])
        # attention masking for padded token

        mask = (
            (x > 0)
            .unsqueeze(1)
            .repeat(sizes=(1, x.shape[1], 1))
            .unsqueeze(1)
            .repeat(sizes=(1, self.attn_heads, 1, 1))
        )

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info)
        sequence_output = [x]
        attn_list = []
        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x,attn = transformer.forward(x, mask)
            sequence_output.append(x)
            attn_list.append(attn)
        
        return x,sequence_output,attn_list
