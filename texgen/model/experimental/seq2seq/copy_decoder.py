from typing import Dict, Optional, Tuple, Union, Callable
import numpy
import torch
from torch import nn
import torch.nn.functional as F
from texar.torch.modules import (WordEmbedder,
                                 PositionEmbedder,
                                 AttentionRNNDecoder,
                                 AttentionRNNDecoderOutput,
                                 TransformerDecoder,
                                 GPT2Decoder)
from texar.torch.modules.decoders.decoder_helpers import (EmbeddingHelper,
                                                          Helper)
from texar.torch.modules.pretrained.gpt2 import PretrainedGPT2Mixin
from texar.torch.modules.decoders.transformer_decoders import TransformerDecoderOutput
from texar.torch.utils.utils import sequence_mask
from texar.torch.core.attention_mechanism_utils import prepare_memory, maybe_mask_score
from texar.torch.utils import transformer_attentions as attn
from texar.torch.utils.shapes import mask_sequences
from torch.nn.parameter import Parameter
from texar.torch.hyperparams import HParams
from ..copy_mechanism import CopyMechanismMixin


# class CopyTransformerDecoderWrapper(PretrainedGPT2Mixin):

#     _IS_DECODE = True

#     def __init__(self, hparams=None):
#         super().__init__(hparams=hparams)

#         self.load_pretrained_config(pretrained_model_name=None,
#                                     cache_dir=None)

#         # Word embedding
#         self.word_embedder = WordEmbedder(
#             vocab_size=self._hparams.vocab_size,
#             hparams=self._hparams.embed)

#         # Position embedding
#         self.position_embedder = PositionEmbedder(
#             position_size=self._hparams.position_size,
#             hparams=self._hparams.position_embed)

#         def embed_fn(tokens, positions):
#             word_embeds = self.word_embedder(tokens)
#             pos_embeds = self.position_embedder(positions)
#             return word_embeds + pos_embeds

#         self.decoder = CopyTransformerDecoder(
#             embed_fn=embed_fn,
#             vocab_size=self._hparams.vocab_size,
#             output_layer=self.word_embedder.embedding,
#             hparams=self._hparams.decoder)

#         self.init_pretrained_weights()

#     @staticmethod
#     def default_hparams():
#         return GPT2Decoder.default_hparams()

#     @property
#     def output_size(self) -> int:
#         return self.decoder.output_size

#     def forward(self, *args, **kwargs):
#         return self.decoder.forward(*args, **kwargs)

# class TransformerLM(CopyMechanismMixin, TransformerDecoder):

#     def __init__(self,
#                  vocab_size: int,
#                  output_layer: Parameter,
#                  embed_fn: Callable,
#                  use_copy_mechanism: bool=False,
#                  hparams: Optional[HParams]=None
#                  ) -> None:
#         TransformerDecoder.__init__(self,
#                                     vocab_size=vocab_size,
#                                     output_layer=output_layer,
#                                     hparams=hparams)
#         self.embed_fn = embed_fn
#         self.use_copy_mechanism = use_copy_mechanism
#         if use_copy_mechanism:
#             # self.enable_copy = True
#             CopyMechanismMixin.__init__(self,
#                                         layer_size=self._output_layer.in_features)

#     def embed_tokens(self,
#                      tokens: torch.LongTensor,
#                      positions: torch.LongTensor
#                      ) -> torch.Tensor:
#         return self.embed_fn(tokens, positions)


class CopyTransformerDecoder(CopyMechanismMixin, TransformerDecoder):

    def __init__(self, embed_fn, vocab_size, output_layer, hparams=None):
        TransformerDecoder.__init__(self,
                                    vocab_size=vocab_size,
                                    output_layer=output_layer,
                                    hparams=hparams)

        self.embed_fn = embed_fn
        # self._enc_dec_attn = nn.Linear(self._output_layer.in_features,
        #                                self._output_layer.in_features)
        # self._dec_with_attn_layer = nn.Linear(self._output_layer.in_features * 2,
        #                                       self._output_layer.in_features)
        # self._gen_prob_layer = nn.Linear(self._output_layer.in_features, 1)
        self.use_copy_mechanism = use_copy_mechanism
        if use_copy_mechanism:
            # self.enable_copy = True
            CopyMechanismMixin.__init__(self,
                                        layer_size=self._output_layer.in_features)
        self.memory_src_ids = None
        self.memory_sequence_length = None

    def embed_tokens(self, tokens: torch.LongTensor,
                     positions: torch.LongTensor) -> torch.Tensor:
        return self.embed_fn(tokens, positions)

    # def apply_attn_and_compute_logits(self, decoder_output, memory):
    #     memory = prepare_memory(memory, self.memory_sequence_length)

    #     attn_scores = torch.matmul(decoder_output,
    #                                self._enc_dec_attn(memory).permute(0, 2, 1))

    #     attn_mask_filter = sequence_mask(
    #         self.memory_sequence_length[:, None].repeat(1, attn_scores.shape[1]),
    #         max_len=attn_scores.shape[-1])
    #     attn_mask_values = torch.tensor(-numpy.inf) * torch.ones_like(attn_scores)
    #     attn_scores = torch.softmax(
    #         torch.where(attn_mask_filter, attn_scores, attn_mask_values),
    #         dim=-1
    #     )

    #     attn_output = torch.bmm(attn_scores, memory)

    #     decoder_with_attn_output = torch.tanh(self._dec_with_attn_layer(
    #         torch.cat([decoder_output, attn_output], dim=-1)
    #     ))

    #     orig_logits = self._output_layer(decoder_output)

    #     p_gen = torch.sigmoid(self._gen_prob_layer(decoder_with_attn_output))
    #     gen_probs = torch.mul(F.softmax(orig_logits, dim=-1),
    #                           p_gen)
    #     copy_probs = torch.mul(attn_scores, 1 - p_gen)
    #     final_logits = torch.log(
    #         gen_probs.scatter_add_(-1,
    #                                self.memory_src_ids[:, None, :].repeat(1, gen_probs.shape[1], 1),
    #                                copy_probs)
    #     )
    #     return final_logits

    def forward(self,
                inputs: Optional[torch.Tensor]=None,
                sequence_length: Optional[torch.LongTensor]=None,
                memory: Optional[torch.Tensor]=None,
                memory_sequence_length: Optional[torch.LongTensor]=None,
                memory_attention_bias: Optional[torch.Tensor]=None,
                context: Optional[torch.Tensor]=None,
                context_sequence_length: Optional[torch.LongTensor]=None,
                helper: Optional[Helper]=None,
                decoding_strategy: str='train_greedy',
                max_decoding_length: Optional[int]=None,
                impute_finished: bool=False,
                infer_mode: Optional[bool]=None,
                beam_width: Optional[int]=None,
                length_penalty: float=0.,
                **kwargs) \
            -> Union[
                TransformerDecoderOutput,
                Tuple[TransformerDecoderOutput, torch.LongTensor],
                Dict[str, torch.Tensor]]:

        memory_src_ids, memory = memory
        self.memory_src_ids = memory_src_ids
        self.memory_sequence_length = memory_sequence_length

        if memory is not None:
            if memory_attention_bias is None:
                if memory_sequence_length is None:
                    raise ValueError(
                        "`memory_sequence_length` is required if "
                        "`memory_attention_bias` is not given.")

                enc_padding = 1 - sequence_mask(
                    memory_sequence_length, memory.size(1),
                    dtype=torch.float32)
                memory_attention_bias = attn.attention_bias_ignore_padding(
                    enc_padding)

        # record the context, which will be used in step function
        # for dynamic_decode
        if context is not None:
            if context_sequence_length is None:
                raise ValueError("'context_sequence_length' must not be None"
                                 "when 'context' is specified.")
            self._state_context = context[:, 1:]
            self._state_context_sequence_length = context_sequence_length - 1
        else:
            self._state_context = None
            self._state_context_sequence_length = None

        # Faster code path for teacher-forcing training
        if (helper is None and beam_width is None and
                decoding_strategy == 'train_greedy'):
            if inputs is None:
                raise ValueError("'input' must not be none "
                                 "when using 'train_greedy' decoding strategy.")
            times = torch.arange(
                inputs.size(1), dtype=torch.long, device=inputs.device)
            times = times.unsqueeze(0).expand(inputs.size(0), -1)
            inputs = self.embed_tokens(inputs, times)
            if sequence_length is not None:
                inputs = mask_sequences(inputs, sequence_length)

            decoder_self_attention_bias = (
                attn.attention_bias_lower_triangle(inputs.size(1)))

            decoder_output = self._self_attention_stack(
                inputs, memory, decoder_self_attention_bias,
                memory_attention_bias, cache=None)

            logits = self.apply_attn_and_compute_logits(decoder_output, memory)

            sample_id = torch.argmax(logits, dim=-1)

            return TransformerDecoderOutput(logits, sample_id)

        # Inference code path.
        if max_decoding_length is None:
            max_decoding_length = self._hparams.max_decoding_length

        self._state_max_decoding_length = max_decoding_length

        if beam_width is None or beam_width == 1:  # Inference-like decoding
            # Prepare helper
            if helper is None:
                kwargs.update(decoding_strategy=decoding_strategy)
                if context is not None:
                    kwargs.update(start_tokens=context[:, 0])
                helper = self._create_or_get_helper(infer_mode, **kwargs)
            assert isinstance(helper, EmbeddingHelper)

            self._state_cache = self._init_cache(
                memory, memory_attention_bias,
                beam_search_decoding=False, batch_size=helper.batch_size)
            if context is not None:
                assert self._state_context is not None
                pad_length = max_decoding_length - self._state_context.size(1)
                if pad_length > 0:
                    self._state_context = torch.cat((
                        self._state_context,
                        self._state_context.new_zeros(
                            self._state_context.size(0), pad_length)
                    ), dim=1)

            outputs, cache, sequence_lengths = self.dynamic_decode(
                helper, inputs=None, sequence_length=None,
                initial_state=None, max_decoding_length=max_decoding_length,
                impute_finished=impute_finished)
            del cache  # not used

            if context is not None:
                # Here the length of sample_id will be larger than that
                # of logit by 1, because there will be a additional
                # start_token in the returned sample_id.
                # the start_id should be the first token of the
                # given context
                start_tokens = context[:, 0]
                outputs = TransformerDecoderOutput(
                    logits=outputs.logits,
                    sample_id=torch.cat([
                        start_tokens.unsqueeze(1),
                        outputs.sample_id
                    ], dim=1))
                sequence_lengths = sequence_lengths + 1

            # Clear caches
            self.memory_src_ids = None
            self.memory_sequence_length = None

            return outputs, sequence_lengths

        else:  # Beam-search decoding
            # Ignore `decoding_strategy` and # assume `helper` is not set.
            if helper is not None:
                raise ValueError("Must not set 'beam_width' and 'helper' "
                                 "simultaneously.")
            if context is not None:
                start_tokens = context[:, 0]
            else:
                if 'start_tokens' not in kwargs:
                    raise ValueError(
                        "'start_tokens' must be specified when using"
                        "beam search decoding.")
                start_tokens = kwargs['start_tokens']
            _batch_size = start_tokens.size(0)
            self._state_cache = self._init_cache(
                memory, memory_attention_bias,
                beam_search_decoding=True,
                batch_size=_batch_size)
            end_token = kwargs.get('end_token')  # type: ignore

            # The output format is different when running beam search.
            sample_id, log_prob = self.beam_decode(
                start_tokens,
                end_token,
                embedding_fn=self.embed_tokens,
                beam_width=beam_width,
                length_penalty=length_penalty,
                decode_length=max_decoding_length)

            # Clear caches
            self.memory_src_ids = None
            self.memory_sequence_length = None

            return {
                'sample_id': sample_id,
                'log_prob': log_prob
            }

    def step(self, helper, time, inputs, state):

        assert state is not None

        decoder_output = self._self_attention_stack(
            inputs.unsqueeze(1), memory=state['memory'], cache=state)

        logits = self.apply_attn_and_compute_logits(decoder_output, state['memory']).squeeze(1)

        sample_ids = helper.sample(time=time, outputs=logits)

        if self._state_context is not None:
            assert self._state_context_sequence_length is not None
            sample_ids = torch.where(
                self._state_context_sequence_length > time,
                self._state_context[:, time],
                sample_ids)

        next_state = state
        outputs = TransformerDecoderOutput(
            logits=logits,
            sample_id=sample_ids)

        return outputs, next_state
