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


class RNNLM(CopyMechanismMixin, BasicRNNDecoder):

    def __init__(self,
                 input_size: int,
                 vocab_size: int,
                 token_embedder: WordEmbedder,
                 use_copy_mechanism: bool=False,
                 hparams: Optional[Dict[str, Any]]=None
                 ) -> None:
        BasicRNNDecoder.__init__(self,
                                 input_size=input_size,
                                 vocab_size=vocab_size,
                                 token_embedder=token_embedder,
                                 hparams=hparams)
        self.use_copy_mechanism = use_copy_mechanism
        if self.use_copy_mechanism:
            CopyMechanismMixin.__init__(self,
                                        layer_size=self._output_layer.in_features)
        self.clear_memory()

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

        if self.use_copy_mechanism:
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
