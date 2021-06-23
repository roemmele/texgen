from typing import Callable, Dict, Optional, Tuple, Union, List, TypeVar
import copy
import numpy
import torch
from torch import nn
import torch.nn.functional as F
import texar.torch as tx
from texar.torch.core.cell_wrappers import HiddenState
from texar.torch.core.attention_mechanism_utils import prepare_memory
from texar.torch.modules import (WordEmbedder,
                                 BasicRNNDecoder,
                                 PositionEmbedder,
                                 TransformerDecoder,
                                 GPT2Decoder,
                                 EmbeddingHelper)
from texar.torch.modules.decoders.decoder_helpers import Helper
from texar.torch.modules.decoders.rnn_decoders import BasicRNNDecoderOutput
from texar.torch.modules.encoders.rnn_encoders import _forward_output_layers
from texar.torch.modules.decoders.transformer_decoders import TransformerDecoderOutput
from texar.torch.modules.pretrained.gpt2 import PretrainedGPT2Mixin
from texar.torch.utils import utils
from texar.torch.utils import transformer_attentions as attn
from texar.torch.utils.utils import sequence_mask
from texar.torch.utils.shapes import mask_sequences
from texar.torch.utils.rnn import dynamic_rnn
from texar.torch.utils.dtypes import torch_bool

State = TypeVar('State')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from texar.torch.data import GPT2Tokenizer
# tokenizer = GPT2Tokenizer('gpt2-small')


def overwrite_state(state, overwriter_state, overwrite_idxs):
    for layer_idx in range(len(state['layers'])):
        if (len(overwriter_state['layers'][layer_idx]['keys']) == 0
                or len(state['layers'][layer_idx]['keys']) == 0):
            state['layers'][layer_idx]['keys'] =\
                copy.copy(overwriter_state['layers'][layer_idx]['keys'])
        else:
            state['layers'][layer_idx]['keys'][0][overwrite_idxs] =\
                overwriter_state['layers'][layer_idx]['keys'][0][overwrite_idxs]
        if (len(overwriter_state['layers'][layer_idx]['values']) == 0
                or len(state['layers'][layer_idx]['values']) == 0):
            state['layers'][layer_idx]['values'] =\
                copy.copy(overwriter_state['layers'][layer_idx]['values'])
        else:
            state['layers'][layer_idx]['values'][0][overwrite_idxs] =\
                overwriter_state['layers'][layer_idx]['values'][0][overwrite_idxs]
    import pdb
    pdb.set_trace()
    return state


class LM(nn.Module):

    def __init__(self,
                 architecture_type,
                 copy_mechanism=None,
                 vocab_size=None,
                 emb_hparams=None,
                 model_hparams=None):
        super().__init__()
        self.architecture_type = architecture_type
        if self.architecture_type == 'rnn':
            token_embedder = WordEmbedder(vocab_size=vocab_size,
                                          hparams=emb_hparams)
            self.decoder = RNNLM(input_size=token_embedder.dim,
                                 vocab_size=token_embedder.vocab_size,
                                 token_embedder=token_embedder,
                                 enable_copy=enable_copy,
                                 hparams=model_hparams)
        elif self.architecture_type == 'transformer':
            self.decoder = TransformerLMWrapper(copy_mechanism=copy_mechanism,
                                                hparams=model_hparams)
        else:
            raise ValueError("architecture_type must be 'rnn' or 'transformer'")

    @property
    def output_size(self) -> int:
        return self.decoder.output_size

    def encode(self, inputs, sequence_length, alignment=None, alt_input_ids=None):
        return self.decoder.encode(inputs, sequence_length, alignment, alt_input_ids)

    def forward(self, batch, decoding_strategy=None,
                helper=None, max_decoding_length=100,
                use_checklist=False, max_checklist_gap=None):
        self.encode(inputs=batch.src_ids,
                    sequence_length=batch.src_lengths,
                    alignment=(batch.tgt_alignment[:, 1:] if 'tgt_alignment'
                               in batch.keys() else None),
                    alt_input_ids=(batch.alt_src_ids if 'alt_src_ids'
                                   in batch.keys() else None))
        # import pdb
        # pdb.set_trace()
        # print(batch)
        if 'tgt_ids' in batch.keys():  # Compute loss
            assert 'tgt_lengths' in batch.keys()
            outputs = self.decoder(inputs=batch.tgt_ids[:, :-1],
                                   sequence_length=batch.tgt_lengths - 1)

            loss = tx.losses.sequence_sparse_softmax_cross_entropy(
                labels=batch.tgt_ids[:, 1:],
                logits=outputs.logits,
                sequence_length=batch.tgt_lengths - 1)
            return {'loss': loss, 'mle_loss': loss}

        else:  # Generate
            outputs, output_lengths = self.decoder(
                start_tokens=torch.full_like(batch.src_ids[:, 0],
                                             batch.bos_token_id),
                end_token=batch.eos_token_id,
                max_decoding_length=max_decoding_length,
                use_checklist=False,  # use_checklist,
                max_checklist_gap=max_checklist_gap)
            return {'pred_ids': outputs.sample_id, 'pred_lengths': output_lengths}

    def set_pg_and_eval_fns(self, pg_fns=[], eval_fns=[]):
        self.pg_fns = pg_fns
        self.eval_fns = eval_fns


class CopyMechanismMixin(nn.Module):

    def __init__(self, layer_size, mechanism_type='standard'):
        self.mechanism_type = mechanism_type
        self.copy_attn = nn.Linear(layer_size, layer_size)
        self.dec_with_copy_attn_layer = nn.Linear(layer_size * 2,
                                                  layer_size)
        self.gen_prob_layer = nn.Linear(layer_size, 1)

    def apply_copy_attn(self, decoder_output, time=None):
        # import pdb
        # pdb.set_trace()
        batch_size = decoder_output.shape[0]

        memory_output = prepare_memory(self.memory_output, self.memory_sequence_length)
        memory_max_length = memory_output.shape[1]

        output_length = decoder_output.shape[1]
        mask_sizes = self.memory_sequence_length[:, None].expand(-1, output_length)

        if self.mechanism_type == 'standard':
            attn_scores = torch.matmul(decoder_output,
                                       self.copy_attn(memory_output).permute(0, 2, 1))
            attn_mask_filter = sequence_mask(
                mask_sizes,
                max_len=memory_max_length
            )
            attn_mask_values = torch.tensor(-numpy.inf) * torch.ones_like(attn_scores)
            attn_scores = torch.softmax(
                torch.where(attn_mask_filter, attn_scores, attn_mask_values),
                dim=-1
            )
        elif self.mechanism_type == 'checklist':
            # import pdb
            # pdb.set_trace()
            if time == None:
                attn_scores = torch.matmul(decoder_output,
                                           self.copy_attn(memory_output).permute(0, 2, 1))
                attn_mask_filter = (~sequence_mask(self.memory_alignment, max_len=memory_max_length)
                                    * (self.memory_alignment > -1)[:, :, None]
                                    )
                attn_mask_filter[:, :, -1] = True
                attn_mask_values = torch.tensor(-numpy.inf) * torch.ones_like(attn_scores)
                attn_scores = torch.softmax(
                    torch.where(attn_mask_filter, attn_scores, attn_mask_values),
                    dim=-1
                )
            else:
                # import pdb
                # pdb.set_trace()
                attn_scores = torch.zeros(batch_size, output_length, memory_max_length,
                                          device=device)
                attn_scores[torch.arange(batch_size, device=device),
                            0,
                            torch.min(self.memory_alignment[:, 0],
                                      self.memory_sequence_length - 1)] = 1.0

        attn_output = torch.bmm(attn_scores, memory_output)

        return attn_scores, attn_output

    def compute_logits_with_copy_attn(self, decoder_output, attn_scores, attn_output):

        decoder_with_attn_output = torch.tanh(self.dec_with_copy_attn_layer(
            torch.cat([decoder_output, attn_output], dim=-1)
        ))

        orig_logits = self._output_layer(decoder_output)

        p_gen = torch.sigmoid(self.gen_prob_layer(decoder_with_attn_output))
        gen_probs = torch.mul(F.softmax(orig_logits, dim=-1),
                              p_gen)

        copy_probs = torch.mul(attn_scores, 1 - p_gen)
        scatter_idxs = self.memory_ids[:, None, :].expand(-1, gen_probs.shape[1], -1)

        final_logits = torch.log(
            gen_probs.scatter_add_(
                -1,
                scatter_idxs,
                copy_probs
            )
        )
        # import pdb
        # pdb.set_trace()
        return final_logits


class RNNLM(CopyMechanismMixin, BasicRNNDecoder):

    def __init__(self, input_size, vocab_size,
                 token_embedder, enable_copy=False, hparams=None):
        BasicRNNDecoder.__init__(self,
                                 input_size=input_size,
                                 vocab_size=vocab_size,
                                 token_embedder=token_embedder,
                                 hparams=hparams)
        self.enable_copy = enable_copy
        if self.enable_copy:
            CopyMechanismMixin.__init__(self,
                                        layer_size=self._output_layer.in_features)
        self.clear_memory()

    def clear_memory(self):
        self.memory_sequence_length = None
        self.memory_ids = None
        self.memory_final_state = None
        self.memory_output = None

    def encode(self,
               inputs: torch.Tensor,
               sequence_length: Optional[Union[torch.LongTensor,
                                               List[int]]]=None):

        self.memory_ids = inputs
        self.memory_sequence_length = sequence_length

        embedded_inputs = self.embed_tokens(tokens=inputs,
                                            positions=None)

        memory_output, memory_final_state = dynamic_rnn(
            cell=self._cell,
            inputs=embedded_inputs,
            sequence_length=sequence_length,
            initial_state=None,
            time_major=False)

        self.memory_final_state = memory_final_state
        self.memory_output = memory_output

    def forward(self, inputs=None, sequence_length=None, start_tokens=None,
                end_token=None, max_decoding_length=None):

        if inputs is not None:
            outputs, _, _ = super()(inputs,
                                    sequence_length,
                                    initial_state=self.memory_final_state,
                                    decoding_strategy='train_greedy')
            self.clear_memory()
            return outputs
        else:
            outputs, _, output_lengths = super()(
                start_tokens=start_tokens,
                end_token=end_token,
                initial_state=self.memory_final_state,
                max_decoding_length=max_decoding_length,
                decoding_strategy='infer_greedy')

            self.clear_memory()
            return outputs, output_lengths

    def step(self,
             helper: Helper,
             time: int,
             inputs: torch.Tensor,
             state: Optional[HiddenState]) \
            -> Tuple[BasicRNNDecoderOutput, HiddenState]:
        cell_outputs, cell_state = self._cell(inputs, state)

        if self.enable_copy:
            cell_outputs = cell_outputs.unsqueeze(1)
            attn_scores, attn_output = self.apply_copy_attn(decoder_output=cell_outputs)
            logits = self.compute_logits_with_copy_attn(decoder_output=cell_outputs,
                                                        attn_scores=attn_scores,
                                                        attn_output=attn_output).squeeze(1)
        else:
            logits = self._output_layer(cell_outputs)

        sample_ids = helper.sample(time=time, outputs=logits)
        next_state = cell_state
        outputs = BasicRNNDecoderOutput(logits, sample_ids, cell_outputs)
        return outputs, next_state


class TransformerLMWrapper(PretrainedGPT2Mixin):

    _IS_DECODE = True

    def __init__(self, copy_mechanism=None, hparams=None):
        super().__init__(hparams=hparams)

        self.load_pretrained_config(pretrained_model_name=None,
                                    cache_dir=None)

        self.word_embedder = WordEmbedder(
            vocab_size=self._hparams.vocab_size,
            hparams=self._hparams.embed)

        self.position_embedder = PositionEmbedder(
            position_size=self._hparams.position_size,
            hparams=self._hparams.position_embed)

        def embed_fn(tokens, positions):
            word_embeds = self.word_embedder(tokens)
            pos_embeds = self.position_embedder(positions)
            return word_embeds + pos_embeds

        self.decoder = TransformerLM(vocab_size=self._hparams.vocab_size,
                                     output_layer=self.word_embedder.embedding,
                                     embed_fn=embed_fn,
                                     copy_mechanism=copy_mechanism,
                                     hparams=self._hparams.decoder)

        self.init_pretrained_weights()

    @staticmethod
    def default_hparams():
        return GPT2Decoder.default_hparams()

    @property
    def output_size(self) -> int:
        return self.decoder.output_size

    def encode(self, inputs, sequence_length,
               alignment=None, alt_input_ids=None):
        return self.decoder.encode(inputs, sequence_length, alignment, alt_input_ids)

    def forward(self, inputs=None, sequence_length=None, start_tokens=None,
                end_token=None, max_decoding_length=None, use_checklist=False,
                max_checklist_gap=None):

        if inputs is not None:
            outputs = self.decoder(inputs,
                                   sequence_length,
                                   decoding_strategy='train_greedy')
            # self.clear_memory()
            return outputs
        else:
            outputs, output_lengths = self.decoder(
                start_tokens=start_tokens,
                end_token=end_token,
                max_decoding_length=max_decoding_length,
                decoding_strategy='infer_greedy',
                use_checklist=use_checklist,
                max_checklist_gap=max_checklist_gap)

            # self.clear_memory()
            return outputs, output_lengths


class TransformerLM(CopyMechanismMixin, TransformerDecoder):

    def __init__(self, vocab_size, output_layer,
                 embed_fn, copy_mechanism=None, hparams=None):
        TransformerDecoder.__init__(self,
                                    vocab_size=vocab_size,
                                    output_layer=output_layer,
                                    hparams=hparams)
        self.embed_fn = embed_fn
        if copy_mechanism is not None:
            self.enable_copy = True
            CopyMechanismMixin.__init__(self,
                                        layer_size=self._output_layer.in_features,
                                        mechanism_type=copy_mechanism)
        else:
            self.enable_copy = False
        self.clear_memory()

    def clear_memory(self):
        self.memory_sequence_length = None
        self.memory_ids = None
        self.memory_output = None
        self.memory_alignment = None
        self.last_aligned_tgt_idx = None
        self.eos_generated = None
        #self.n_unchecked_steps = None

    def embed_tokens(self, tokens: torch.LongTensor,
                     positions: torch.LongTensor) -> torch.Tensor:
        return self.embed_fn(tokens, positions)

    def encode(self,
               inputs: Union[torch.Tensor, torch.LongTensor],
               sequence_length: Optional[torch.LongTensor]=None,
               alignment=None,
               alt_input_ids=None):

        self.memory_ids = inputs
        self.memory_sequence_length = sequence_length
        self.memory_alignment = alignment
        self.alt_memory_ids = alt_input_ids
        self.all_memory_ids = torch.cat([self.memory_ids[:, None],
                                         self.alt_memory_ids], dim=1)

        times = torch.arange(
            inputs.size(1), dtype=torch.long, device=inputs.device)
        times = times.unsqueeze(0).expand(inputs.size(0), -1)
        inputs = self.embed_tokens(inputs, times)
        if sequence_length is not None:
            inputs = mask_sequences(inputs, sequence_length)

        self_attention_bias = (
            attn.attention_bias_lower_triangle(inputs.size(1)))

        output = self._self_attention_stack(
            inputs, memory=None,
            decoder_self_attention_bias=self_attention_bias,
            memory_attention_bias=None, cache=None)

        self.memory_output = output

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
                use_checklist=False,
                max_checklist_gap: Optional[int]=None,
                **kwargs) \
            -> Union[
                TransformerDecoderOutput,
                Tuple[TransformerDecoderOutput, torch.LongTensor],
                Dict[str, torch.Tensor]]:
        '''This function was largely copied from the Texar TransformerDecoder.forward() method.
        https://texar-pytorch.readthedocs.io/en/latest/_modules/texar/torch/modules/decoders/transformer_decoders.html#TransformerDecoder'''

        # memory_ids, memory = memory
        # self.use_checklist = use_checklist
        memory = self.memory_output
        memory_sequence_length = self.memory_sequence_length
        # self.memory_ids = memory_ids
        # self.memory_sequence_length = memory_sequence_length

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

            # logits = self.apply_attn_and_compute_logits(decoder_output, memory)
            if self.enable_copy:
                attn_scores, attn_output = self.apply_copy_attn(decoder_output)
                logits = self.compute_logits_with_copy_attn(decoder_output,
                                                            attn_scores,
                                                            attn_output)
            else:
                logits = self._output_layer(decoder_output)

            sample_id = torch.argmax(logits, dim=-1)

            self.clear_memory()

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
                impute_finished=impute_finished, use_checklist=use_checklist,
                max_checklist_gap=max_checklist_gap)
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

            self.clear_memory()

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

            self.clear_memory()

            return {
                'sample_id': sample_id,
                'log_prob': log_prob
            }

    def step(self, helper, time, inputs, state, end_token,
             use_checklist=False, max_checklist_gap=None):

        assert state is not None

        decoder_output = self._self_attention_stack(
            inputs.unsqueeze(1), memory=state['memory'], cache=state)

        if self.enable_copy:
            attn_scores, attn_output = self.apply_copy_attn(decoder_output, time=time)
            logits = self.compute_logits_with_copy_attn(decoder_output,
                                                        attn_scores,
                                                        attn_output).squeeze(1)
        else:
            logits = self._output_layer(decoder_output)

        logits = logits.squeeze(1)
        sample_ids = helper.sample(time=time, outputs=logits)

        if use_checklist:

            (sample_ids,
             state,
             output_time) = self.checklist_step(time=time,
                                                sample_ids=sample_ids,
                                                state=state,
                                                end_token=end_token,
                                                max_checklist_gap=max_checklist_gap)
        else:
            output_time = time + 1

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

        return outputs, next_state, output_time

    def checklist_step(self, time, sample_ids, state,
                       end_token=None,
                       max_checklist_gap=None):

        import pdb
        pdb.set_trace()

        print("original sample:\t", sample_ids)

        if time == -1:  # Initialize
            self.memory_alignment = torch.zeros(*sample_ids.shape,
                                                device=device,
                                                dtype=torch.long)  # [:, None]
            self.last_aligned_tgt_idx = torch.ones(*sample_ids.shape,
                                                   device=device,
                                                   dtype=torch.long) * time  # [:, None]
            self.eos_generated = torch.zeros(*sample_ids.shape,
                                             device=device,
                                             dtype=torch.bool)
            self.last_aligned_state = copy.deepcopy(state)
        else:
            self.eos_generated[~self.eos_generated] = (
                sample_ids == end_token)[~self.eos_generated]

        output_time = torch.ones(*sample_ids.shape,
                                 device=device,
                                 dtype=torch.long) * time + 1

        if max_checklist_gap is None:
            max_checklist_gap = self.memory_ids.shape[-1] + 1
        n_unchecked_steps = time - self.last_aligned_tgt_idx
        gap_exceeded = n_unchecked_steps >= max_checklist_gap
        # unchecked_memory_after_eos = (self.eos_generated &
        #                               (self.memory_sequence_length
        #                                - (self.memory_alignment + 1) > 0))
        unchecked_memory_after_eos = (self.eos_generated &
                                      (self.memory_sequence_length -
                                       self.memory_alignment > 0))
        do_overwrite = gap_exceeded | unchecked_memory_after_eos
        sample_ids[do_overwrite] = self.memory_ids[
            torch.arange(self.memory_ids.shape[0],
                         device=device),
            self.memory_alignment][do_overwrite]

        if do_overwrite.sum() > 0:
            state = overwrite_state(
                state=state,
                overwriter_state=self.last_aligned_state,
                overwrite_idxs=do_overwrite.nonzero()[:, 0]
            )

        output_time[do_overwrite] = self.last_aligned_tgt_idx[do_overwrite] + 1
        self.eos_generated[do_overwrite] = False

        batch_size = sample_ids.shape[0]

        sample_in_memory = self.all_memory_ids[torch.arange(batch_size,
                                                            device=device),
                                               :, self.memory_alignment] == sample_ids[:, None]
        sample_in_memory = torch.any(sample_in_memory, dim=1)
        alignment_needs_update = sample_in_memory & ~self.eos_generated

        # self.memory_alignment[alignment_needs_update] = torch.min(
        #     (self.memory_sequence_length[alignment_needs_update] - 1),
        #     1 + self.memory_alignment[alignment_needs_update]
        # )
        if alignment_needs_update.sum() > 0:
            self.last_aligned_state = overwrite_state(
                state=self.last_aligned_state,
                overwriter_state=state,
                overwrite_idxs=alignment_needs_update.nonzero()[:, 0]
            )

        self.memory_alignment[alignment_needs_update] += 1
        self.last_aligned_tgt_idx[alignment_needs_update
                                  & ~self.eos_generated] = output_time

        print("overwritten sample:\t", sample_ids)
        print("eos generated:\t", self.eos_generated)
        print("gap exceeded:\t", gap_exceeded)
        print("unchecked memory after eos:\t", unchecked_memory_after_eos)
        print("alignment needs update:\t", alignment_needs_update)
        print("n unchecked_steps:\t", n_unchecked_steps)
        print("memory sequence length:\t", self.memory_sequence_length)
        print("memory alignment:\t", self.memory_alignment)
        print("last aligned target:\t", self.last_aligned_tgt_idx)
        print("output time\t", output_time)

        return sample_ids, state, output_time

    def dynamic_decode(self, helper: Helper, inputs: Optional[torch.Tensor],
                       sequence_length: Optional[torch.LongTensor],
                       initial_state: Optional[State],
                       max_decoding_length: Optional[int]=None,
                       impute_finished: bool=False,
                       step_hook: Optional[Callable[[int], None]]=None,
                       use_checklist=False,
                       max_checklist_gap=None
                       ) -> Tuple[TransformerDecoderOutput, Optional[State], torch.LongTensor]:
        """Generic routine for dynamic decoding. Please check the
        `documentation
        <https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/dynamic_decode>`_
        for the TensorFlow counterpart.

        Returns:
            A tuple of output, final state, and sequence lengths. Note that
            final state could be `None`, when all sequences are of zero length
            and :attr:`initial_state` is also `None`.
        """

        # Decode
        finished, step_inputs, state = self.initialize(
            helper, inputs, sequence_length, initial_state)

        zero_outputs = step_inputs.new_zeros(
            step_inputs.size(0), self.output_size)

        if max_decoding_length is not None:
            finished |= (max_decoding_length <= 0)
        sequence_lengths = torch.zeros_like(
            finished, dtype=torch.long, device=finished.device)

        if use_checklist:
            self.checklist_step(time=-1, sample_ids=helper._start_tokens,
                                state=state)
        #time = 0
        iter_time = 0

        outputs = []
        omit_outputs = []
        # import pdb
        # pdb.set_trace()
        # outputs = torch.ones(step_inputs.shape[0], max_decoding_length,
        #                      dtype=torch.long, device=device) * helper._end_token

        while (not torch.all(finished).item() and
               (max_decoding_length is None or iter_time < max_decoding_length)):

            # if (max_decoding_length - iter_time) <= torch.max(self.memory_sequence_length
            #                                                   - self.memory_alignment):
            #     max_checklist_gap = 0
            import pdb
            pdb.set_trace()

            (next_outputs,
             decoder_state,
             output_time) = self.step(helper, iter_time, step_inputs, state,
                                      end_token=helper._end_token,
                                      use_checklist=use_checklist,
                                      max_checklist_gap=max_checklist_gap)

            if max_decoding_length is not None and \
                    iter_time + 1 == max_decoding_length:
                # Maximum decoding length reached, mark all batches as finished.
                # This requires special handling because performing lookup on
                # position embeddings with `time + 1` may result in IndexError.
                decoder_finished = torch.tensor(1, dtype=torch_bool,
                                                device=finished.device)
                # Since `next_inputs` will not be used, simply create a null
                # tensor.
                next_inputs = torch.empty(0)
            else:
                next_inputs, decoder_finished = self.next_inputs(
                    helper, iter_time, next_outputs)

            if getattr(self, 'tracks_own_finished', False):
                next_finished = decoder_finished
            else:
                next_finished = decoder_finished | finished

            # Zero out output values past finish
            if impute_finished:
                emit = utils.map_structure_zip(
                    lambda new, cur: torch.where(finished, cur, new),
                    (next_outputs, zero_outputs))
                next_state = utils.map_structure_zip(
                    lambda new, cur: torch.where(finished, cur, new),
                    (decoder_state, state))
            else:
                emit = next_outputs
                next_state = decoder_state

            outputs.append(emit)
            omit_outputs.append(output_time < (iter_time - 1))
            #outputs[:, output_time] = emit
            sequence_lengths.index_fill_(
                dim=0, value=iter_time + 1,
                index=torch.nonzero((~finished).long()).flatten())
            # sequence_lengths.index_fill_(
            #     dim=0, value=output_time + 1,
            #     index=torch.nonzero((~finished).long()).flatten())
            iter_time += 1
            finished = next_finished
            step_inputs = next_inputs
            state = next_state

            if step_hook is not None:
                step_hook(time)

        final_outputs = utils.map_structure_zip(
            lambda *tensors: torch.stack(tensors),
            outputs)  # output at each time step may be a namedtuple
        final_state = state
        final_sequence_lengths = sequence_lengths

        try:
            final_outputs, final_state = self.finalize(
                final_outputs, final_state, final_sequence_lengths)
        except NotImplementedError:
            pass

        if not self._output_time_major:
            final_outputs = utils.map_structure(
                lambda x: x.transpose(0, 1) if x.dim() >= 2 else x,
                final_outputs)

        # return final_outputs, final_state, final_sequence_lengths

        # (final_outputs,
        #  final_state,
        #  final_sequence_lengths) = super().dynamic_decode(*args, **kwargs)

        # if use_checklist:
        #     # import pdb
        #     # pdb.set_trace()
        #     batch_size = final_outputs.sample_id.shape[0]
        #     end_token = helper._end_token  # args[0] is helper

        #     all_new_sample_ids = torch.ones_like(final_outputs.sample_id) * end_token
        #     all_new_sample_ids[:, :final_outputs.sample_id.shape[-1]] = final_outputs.sample_id

        #     for idx, (sample_ids,
        #               last_aligned_tgt,
        #               memory_ids, memory_length,
        #               memory_alignment) in enumerate(zip(final_outputs.sample_id,
        #                                                  self.last_aligned_tgt_idx,
        #                                                  self.memory_ids,
        #                                                  self.memory_sequence_length,
        #                                                  self.memory_alignment)):
        #         # print(tokenizer.map_id_to_text(sample_ids.numpy()))
        #         # if idx == 25:
        #         #     import pdb
        #         #     pdb.set_trace()
        #         n_unsatisfied_tokens = (memory_length
        #                                 - (memory_alignment + 1))
        #         if n_unsatisfied_tokens > 0:
        #             # import pdb
        #             # pdb.set_trace()
        #             appended_memory_ids = memory_ids[memory_alignment:memory_length]
        #             new_sample_ids = torch.cat([sample_ids[:last_aligned_tgt],
        #                                         appended_memory_ids[:-1],
        #                                         end_token.expand(1)])
        #             new_sample_ids = new_sample_ids[:max_decoding_length]
        #             new_sample_length = new_sample_ids.shape[-1]
        #             if new_sample_length > all_new_sample_ids.shape[-1]:  # Expand sample matrix
        #                 all_new_sample_ids = torch.cat(
        #                     [all_new_sample_ids,
        #                      torch.ones(batch_size,
        #                                 new_sample_length - all_new_sample_ids.shape[-1],
        #                                 device=device,
        #                                 dtype=torch.long) * end_token],
        #                     dim=-1
        #                 )
        #             all_new_sample_ids[idx, :new_sample_ids.shape[-1]] = new_sample_ids
        #             final_sequence_lengths[idx] = new_sample_ids.shape[-1]

        #     final_outputs = TransformerDecoderOutput(final_outputs.logits,
        #                                              all_new_sample_ids)

        return final_outputs, final_state, final_sequence_lengths
