from torch import Tensor, nn
from texar.torch.modules import (WordEmbedder,
                                 UnidirectionalRNNEncoder,
                                 GPT2Encoder)
from texar.torch.data.data.dataset_utils import Batch
from typing import Dict, Optional, Union, Any


class RNNEncoderWrapper(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 embedder_hparams: Optional[Dict[str, int]]=None,
                 model_hparams: Optional[Dict[str, Any]]=None
                 ) -> None:
        super().__init__()
        self.token_embedder = WordEmbedder(vocab_size=vocab_size,
                                           hparams=embedder_hparams)
        self.encoder = UnidirectionalRNNEncoder(input_size=self.token_embedder.dim,
                                                hparams=model_hparams)
        self.output_size = self.encoder.output_size

    def __call__(self,
                 batch: Batch):
        encoder_inputs = self.token_embedder(batch.src_ids)
        enc_states = self.encoder(inputs=encoder_inputs,
                                  sequence_length=batch.src_lengths)[0]
        return enc_states


class TransformerEncoderWrapper(nn.Module):

    def __init__(self,
                 model_hparams: Optional[Dict[str, Any]]=None
                 ) -> None:
        super().__init__()
        # Note that Transformer encoder uses GPT2Encoder architecture, but weights are randomly initialized,
        # rather than using pretrained weights. This is all that distinguishes it from the GPT2Encoder.
        self.encoder = GPT2Encoder(hparams=model_hparams)
        self.output_size = self.encoder.output_size

    def __call__(self,
                 batch: Batch
                 ) -> Tensor:
        encoder_inputs = batch.src_ids
        enc_states = self.encoder(inputs=encoder_inputs,
                                  sequence_length=batch.src_lengths)
        return enc_states
