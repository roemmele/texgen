from typing import Dict, Optional, Tuple, Union, Callable
import numpy
import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from texar.torch.modules import (WordEmbedder,
                                 PositionEmbedder,
                                 AttentionRNNDecoder,
                                 AttentionRNNDecoderOutput,
                                 TransformerDecoder,
                                 TransformerDecoderOutput,
                                 GPT2Decoder,
                                 PretrainedGPT2Mixin,
                                 Helper)
from texar.torch.utils.utils import sequence_mask
from texar.torch.core.attention_mechanism_utils import prepare_memory, maybe_mask_score
from texar.torch.utils import transformer_attentions as attn
from texar.torch.utils.shapes import mask_sequences
from texar.torch.hyperparams import HParams
from texar.torch.core.attention_mechanism import AttentionWrapperState


class CopyAttentionRNNDecoder(AttentionRNNDecoder):

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._gen_prob_layer = nn.Linear(self._output_layer.in_features, 1)

    def step(self,
             helper: Helper,
             time: int,
             inputs: Tensor,
             state: Optional[AttentionWrapperState]
             ) -> Tuple[AttentionRNNDecoderOutput, AttentionWrapperState]:
        memory_src_ids, memory_states = self.memory
        wrapper_outputs, wrapper_state = self._cell(
            inputs, state, memory_states, self.memory_sequence_length)

        attention_scores = wrapper_state.alignments
        attention_context = wrapper_state.attention

        orig_logits = self._output_layer(wrapper_outputs)
        p_gen = torch.sigmoid(self._gen_prob_layer(wrapper_outputs))
        gen_probs = torch.mul(F.softmax(orig_logits, dim=1),
                              p_gen)
        copy_probs = torch.mul(attention_scores, 1 - p_gen)
        final_logits = torch.log(gen_probs.scatter_add_(1, memory_src_ids, copy_probs))
        sample_ids = helper.sample(time=time, outputs=final_logits)

        outputs = AttentionRNNDecoderOutput(
            final_logits, sample_ids, wrapper_outputs,
            attention_scores, attention_context)
        next_state = wrapper_state

        return outputs, next_state
