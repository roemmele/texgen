import numpy
import torch
from torch import nn
from texar.torch.core.attention_mechanism_utils import prepare_memory
from texar.torch.utils.utils import sequence_mask


class RNNCopyMechanismMixin(nn.Module):

    def __init__(self,
                 layer_size: int):
        self.copy_attn = nn.Linear(layer_size, layer_size)
        self.dec_with_copy_attn_layer = nn.Linear(layer_size * 2,
                                                  layer_size)
        self.gen_prob_layer = nn.Linear(layer_size, 1)

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


class TransformerCopyMechanismMixin(nn.Module):

    def __init__(self,
                 layer_size: int):
        self.copy_attn = nn.Linear(layer_size, layer_size)
        self.dec_with_copy_attn_layer = nn.Linear(layer_size * 2,
                                                  layer_size)
        self.gen_prob_layer = nn.Linear(layer_size, 1)

    def apply_copy_attn(self,
                        decoder_output,
                        time=None):

        batch_size = decoder_output.shape[0]

        memory_output = prepare_memory(self.memory_output, self.memory_sequence_length)
        memory_max_length = memory_output.shape[1]

        output_length = decoder_output.shape[1]
        mask_sizes = self.memory_sequence_length[:, None].expand(-1, output_length)

        attn_scores = torch.matmul(decoder_output,
                                   self.copy_attn(memory_output).permute(0, 2, 1))
        attn_mask_filter = sequence_mask(
            mask_sizes,
            max_len=memory_max_length
        )
        attn_mask_values = torch.tensor(numpy.inf) * torch.ones_like(attn_scores)
        attn_scores = torch.softmax(
            torch.where(attn_mask_filter, attn_scores, attn_mask_values),
            dim=-1
        )

        attn_output = torch.bmm(attn_scores, memory_output)

        return attn_scores, attn_output

    def compute_logits_with_copy_attn(self,
                                      decoder_output,
                                      attn_scores,
                                      attn_output):
        decoder_with_attn_output = torch.tanh(self.dec_with_copy_attn_layer(
            torch.cat([decoder_output, attn_output], dim=-1)
        ))

        orig_logits = self._output_layer(decoder_output)

        p_gen = torch.sigmoid(self.gen_prob_layer(decoder_with_attn_output))
        gen_probs = torch.mul(nn.functional.softmax(orig_logits, dim=-1),
                              p_gen)

        import pdb
        pdb.set_trace()
        copy_probs = torch.mul(attn_scores, 1 - p_gen)
        scatter_idxs = self.memory_ids[:, None, :].expand(-1, gen_probs.shape[1], -1)

        final_logits = torch.log(
            gen_probs.scatter_add_(
                -1,
                scatter_idxs,
                copy_probs
            )
        )
        return final_logits
