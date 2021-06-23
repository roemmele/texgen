from typing import Any, Callable, Dict, Optional, Tuple, Union, List
import torch
from torch import LongTensor, Tensor, nn
from torch.nn.parameter import Parameter
import texar.torch as tx
from texar.torch.core.cell_wrappers import HiddenState
from texar.torch.modules import (WordEmbedder,
                                 BasicRNNDecoder,
                                 BasicRNNDecoderOutput,
                                 PositionEmbedder,
                                 TransformerDecoder,
                                 TransformerDecoderOutput,
                                 Helper,
                                 EmbeddingHelper,
                                 GreedyEmbeddingHelper,
                                 GPT2Decoder,
                                 PretrainedGPT2Mixin)
from texar.torch.utils import transformer_attentions as attn
from texar.torch.utils.shapes import mask_sequences
from texar.torch.utils.rnn import dynamic_rnn
from texar.torch.data.data.dataset_utils import Batch
from texar.torch.hyperparams import HParams

from .generation_utils import get_generation_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LM(nn.Module):

    def __init__(self,
                 architecture_type: str,
                 vocab_size: Optional[str]=None,
                 emb_hparams: Optional[Dict]=None,
                 model_hparams: Optional[Dict[str, Any]]=None
                 ) -> None:
        super().__init__()
        self.architecture_type = architecture_type
        if self.architecture_type == 'rnn':
            token_embedder = WordEmbedder(vocab_size=vocab_size,
                                          hparams=emb_hparams)
            self.decoder = RNNLM(input_size=token_embedder.dim,
                                 vocab_size=token_embedder.vocab_size,
                                 token_embedder=token_embedder,
                                 hparams=model_hparams)
        elif self.architecture_type == 'transformer':
            self.decoder = TransformerLMWrapper(hparams=model_hparams)
        else:
            raise ValueError("architecture_type must be 'rnn' or 'transformer'")

    @property
    def output_size(self) -> int:
        return self.decoder.output_size

    def encode(self,
               inputs: Tensor,
               sequence_length: Tensor
               ) -> Tensor:
        return self.decoder.encode(inputs, sequence_length)

    def forward(self,
                batch: Batch,
                max_decoding_length: int=100,
                infer_method: str='greedy',
                sample_top_k: int=0,
                sample_p: float=1.0,
                sample_temperature: float=1.0
                ) -> Dict[str, Tensor]:

        memory_output = self.encode(inputs=batch.src_ids,
                                    sequence_length=batch.src_lengths)

        if 'tgt_ids' in batch.keys():  # Compute loss
            assert 'tgt_lengths' in batch.keys()

            outputs = self.decoder(inputs=batch.tgt_ids[:, :-1],
                                   sequence_length=batch.tgt_lengths - 1,
                                   memory=memory_output,
                                   memory_sequence_length=batch.src_lengths)

            loss = tx.losses.sequence_sparse_softmax_cross_entropy(
                labels=batch.tgt_ids[:, 1:],
                logits=outputs.logits,
                sequence_length=batch.tgt_lengths - 1)

            return {'loss': loss, 'mle_loss': loss}

        else:  # Generate
            generation_fn = get_generation_fn(
                start_tokens=torch.full_like(
                    batch.src_ids[:, 0],
                    (batch.ctx_ids[0, 0] if 'ctx_ids' in batch.keys()
                     else batch.bos_token_id)),
                end_token=batch.eos_token_id,
                infer_method=infer_method,
                sample_top_k=sample_top_k,
                sample_p=sample_p,
                sample_temperature=sample_temperature
            )
            outputs, output_lengths = self.decoder(
                memory=memory_output,
                memory_sequence_length=batch.src_lengths,
                context=batch.ctx_ids if 'ctx_ids' in batch.keys() else None,
                context_sequence_length=(batch.ctx_lengths
                                         if 'ctx_lengths' in batch.keys() else None),
                helper=generation_fn,
                max_decoding_length=max_decoding_length)
            return {'pred_ids': outputs.sample_id, 'pred_lengths': output_lengths}

    def set_pg_and_eval_fns(self,
                            pg_fns: List[Any]=[],
                            eval_fns: List[Any]=[]
                            ) -> None:
        self.pg_fns = pg_fns
        self.eval_fns = eval_fns


class RNNLM(BasicRNNDecoder):

    def __init__(self,
                 input_size: int,
                 vocab_size: int,
                 token_embedder: WordEmbedder,
                 hparams: Optional[Dict[str, Any]]=None
                 ) -> None:
        BasicRNNDecoder.__init__(self,
                                 input_size=input_size,
                                 vocab_size=vocab_size,
                                 token_embedder=token_embedder,
                                 hparams=hparams)

    def clear_memory(self) -> None:
        self.memory_sequence_length = None
        self.memory_ids = None
        self.memory_final_state = None
        self.memory_output = None

    def encode(self,
               inputs: torch.Tensor,
               sequence_length: Optional[Union[torch.LongTensor,
                                               List[int]]]=None
               ) -> None:

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

    def forward(self,
                inputs: Optional[Tensor]=None,
                sequence_length: Optional[Tensor]=None,
                memory: Optional[Tensor]=None,
                memory_sequence_length: Optional[Tensor]=None,
                context: Optional[Tensor]=None,
                context_sequence_length: Optional[Tensor]=None,
                helper: Optional[Helper]=None,
                start_tokens: Optional[LongTensor]=None,
                end_token: Union[int, LongTensor]=None,
                max_decoding_length: Optional[int]=None
                ) -> Union[Tuple[BasicRNNDecoderOutput, Tensor], BasicRNNDecoderOutput]:

        if inputs is not None:
            outputs, _, _ = super().forward(
                inputs=inputs,
                sequence_length=sequence_length,
                decoding_strategy='train_greedy')
            self.clear_memory()
            return outputs
        else:
            outputs, _, output_lengths = super().forward(
                memory=memory,
                memory_sequence_length=memory_sequence_length,
                initial_state=self.memory_final_state,
                helper=helper,
                start_tokens=start_tokens,
                end_token=end_token,
                max_decoding_length=max_decoding_length,
                decoding_strategy='infer_greedy')
            self.clear_memory()
            return outputs, output_lengths

    def step(self,
             helper: Helper,
             time: int,
             inputs: torch.Tensor,
             state: Optional[HiddenState]
             ) -> Tuple[BasicRNNDecoderOutput, HiddenState]:
        cell_outputs, cell_state = self._cell(inputs, state)

        logits = self._output_layer(cell_outputs)

        sample_ids = helper.sample(time=time, outputs=logits)
        next_state = cell_state
        outputs = BasicRNNDecoderOutput(logits, sample_ids, cell_outputs)
        return outputs, next_state


class TransformerLMWrapper(PretrainedGPT2Mixin):

    _IS_DECODE = True

    def __init__(self,
                 hparams: Optional[Dict[str, Any]]=None
                 ) -> None:
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
                                     hparams=self._hparams.decoder)

        self.init_pretrained_weights()

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        return GPT2Decoder.default_hparams()

    @property
    def output_size(self) -> int:
        return self.decoder.output_size

    def encode(self,
               inputs: Tensor,
               sequence_length: Tensor
               ) -> Tensor:
        return self.decoder.encode(inputs, sequence_length)

    def forward(self,
                inputs: Optional[Tensor]=None,
                sequence_length: Optional[Tensor]=None,
                memory: Optional[Tensor]=None,
                memory_sequence_length: Optional[Tensor]=None,
                context: Optional[Tensor]=None,
                context_sequence_length: Optional[Tensor]=None,
                helper: Optional[GreedyEmbeddingHelper]=None,
                start_tokens: Optional[LongTensor]=None,
                end_token: Union[int, LongTensor]=None,
                max_decoding_length: Optional[int]=None
                ) -> Union[TransformerDecoderOutput, Tuple[TransformerDecoderOutput, Tensor]]:

        if inputs is not None:
            outputs = self.decoder(inputs=inputs,
                                   sequence_length=sequence_length,
                                   memory=memory,
                                   memory_sequence_length=memory_sequence_length,
                                   decoding_strategy='train_greedy')
            return outputs
        else:
            outputs, output_lengths = self.decoder(
                memory=memory,
                memory_sequence_length=memory_sequence_length,
                context=context,
                context_sequence_length=context_sequence_length,
                helper=helper,
                start_tokens=start_tokens,
                end_token=end_token,
                max_decoding_length=max_decoding_length,
                decoding_strategy='infer_greedy')
            return outputs, output_lengths


class TransformerLM(TransformerDecoder):

    def __init__(self,
                 vocab_size: int,
                 output_layer: Parameter,
                 embed_fn: Callable,
                 hparams: Optional[HParams]=None
                 ) -> None:
        TransformerDecoder.__init__(self,
                                    vocab_size=vocab_size,
                                    output_layer=output_layer,
                                    hparams=hparams)
        self.embed_fn = embed_fn

    def embed_tokens(self,
                     tokens: torch.LongTensor,
                     positions: torch.LongTensor
                     ) -> torch.Tensor:
        return self.embed_fn(tokens, positions)

    def encode(self,
               inputs: Union[torch.Tensor, torch.LongTensor],
               sequence_length: Optional[torch.LongTensor]=None
               ) -> Tensor:

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

        return output
