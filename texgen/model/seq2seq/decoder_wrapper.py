from typing import Dict, Optional, Tuple, Union, List, Any
import torch
from torch import nn, Tensor, LongTensor
import texar.torch as tx
from texar.torch.modules import (WordEmbedder,
                                 AttentionRNNDecoder,
                                 AttentionRNNDecoderOutput,
                                 TransformerDecoderOutput,
                                 Helper,
                                 GPT2Decoder)

from .copy_decoder import CopyAttentionRNNDecoder
from ..generation_utils import get_generation_fn


class TransformerDecoderWrapper(nn.Module):

    def __init__(self,
                 model_hparams: Optional[Dict[str, Any]]=None
                 ) -> None:
        super().__init__()
        # Note that Transformer encoder uses GPT2Decoder architecture, but weights are randomly initialized,
        # rather than using pretrained weights. This is all that distinguishes it from the GPT2Decoder.
        self.decoder = GPT2Decoder(hparams=model_hparams)

    @property
    def output_size(self) -> int:
        return self.decoder.output_size

    def forward(self,
                inputs: Optional[Tensor]=None,
                sequence_length: Optional[Tensor]=None,
                memory_ids: Optional[Tensor]=None,
                memory: Optional[Tensor]=None,
                memory_sequence_length: Optional[Tensor]=None,
                context: Optional[Tensor]=None,
                context_sequence_length: Optional[Tensor]=None,
                helper: Optional[Helper]=None,
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


class RNNDecoderWrapper(nn.Module):

    def __init__(self,
                 encoder_output_size: int,
                 use_copy_mechanism: bool=False,
                 token_embedder: Optional[WordEmbedder]=None,
                 vocab_size: Optional[int]=None,
                 embedder_hparams: Optional[Dict[str, int]]=None,
                 model_hparams: Optional[Dict[str, Any]]=None) -> None:
        super().__init__()
        self.token_embedder = token_embedder

        if not self.token_embedder:
            assert vocab_size is not None and embedder_hparams is not None
            self.token_embedder = WordEmbedder(vocab_size=vocab_size,
                                               hparams=embedder_hparams)

        self.use_copy_mechanism = use_copy_mechanism
        if self.use_copy_mechanism:
            decoder_cls = CopyAttentionRNNDecoder
        else:
            decoder_cls = AttentionRNNDecoder
        self.decoder = decoder_cls(input_size=self.token_embedder.dim,
                                   encoder_output_size=encoder_output_size,
                                   vocab_size=self.token_embedder.vocab_size,
                                   token_embedder=self.token_embedder,
                                   hparams=model_hparams)

    @property
    def output_size(self) -> int:
        return self.decoder.output_size

    def forward(self,
                inputs: Optional[Tensor]=None,
                sequence_length: Optional[Tensor]=None,
                memory_ids: Optional[Tensor]=None,
                memory: Optional[Tensor]=None,
                memory_sequence_length: Optional[Tensor]=None,
                context: Optional[Tensor]=None,
                context_sequence_length: Optional[Tensor]=None,
                helper: Optional[Helper]=None,
                start_tokens: Optional[LongTensor]=None,
                end_token: Union[int, LongTensor]=None,
                max_decoding_length: Optional[int]=None
                ) -> Union[Tuple[AttentionRNNDecoderOutput, Tensor], AttentionRNNDecoderOutput]:
        if self.use_copy_mechanism:
            memory = (memory_ids, memory)

        if inputs is not None:
            outputs, _, _ = self.decoder(inputs=inputs,
                                         sequence_length=sequence_length,
                                         memory=memory,
                                         memory_sequence_length=memory_sequence_length,
                                         decoding_strategy='train_greedy')
            return outputs
        else:
            outputs, _, output_lengths = self.decoder(
                memory=memory,
                memory_sequence_length=memory_sequence_length,
                helper=helper,
                start_tokens=start_tokens,
                end_token=end_token,
                max_decoding_length=max_decoding_length,
                decoding_strategy='infer_greedy')
            return outputs, output_lengths
