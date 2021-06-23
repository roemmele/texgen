import importlib
import logging
import pickle
import os
import numpy
from typing import Any, List, Dict, Optional, Tuple, Union
import torch
from torch import nn, LongTensor, Tensor
import torch.nn.functional as F
import texar.torch as tx
from texar.torch.data.data.dataset_utils import Batch
from .encoder_wrapper import RNNEncoderWrapper, TransformerEncoderWrapper
from .decoder_wrapper import RNNDecoderWrapper, TransformerDecoderWrapper
from ..generation_utils import get_generation_fn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


class Seq2SeqModel(nn.Module):

    def __init__(self,
                 encoder_type: str,
                 decoder_type: str,
                 use_copy_mechanism: bool=False,
                 vocab_size: int=None,
                 embedder_hparams: Optional[Dict[str, int]]=None,
                 encoder_hparams: Optional[Dict[str, Any]]=None,
                 decoder_hparams: Optional[Dict[str, Any]]=None
                 ) -> None:
        super().__init__()
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type

        assert self.encoder_type in ('rnn', 'transformer')
        if self.encoder_type == 'rnn':
            self.encoder = RNNEncoderWrapper(vocab_size=vocab_size,
                                             embedder_hparams=embedder_hparams,
                                             model_hparams=encoder_hparams)
        elif self.encoder_type == 'transformer':
            self.encoder = TransformerEncoderWrapper(model_hparams=encoder_hparams)

        assert self.decoder_type in ('rnn', 'transformer')
        if self.decoder_type == 'rnn':
            self.decoder = RNNDecoderWrapper(
                encoder_output_size=self.encoder.output_size,
                use_copy_mechanism=use_copy_mechanism,
                token_embedder=(self.encoder.token_embedder if
                                self.encoder_type == 'rnn' else None),
                vocab_size=vocab_size,
                embedder_hparams=embedder_hparams,
                model_hparams=decoder_hparams
            )

        elif self.decoder_type == 'transformer':
            assert not use_copy_mechanism, "Copy mechanism currently not implemented for transformer layers, only RNN"
            self.decoder = TransformerDecoderWrapper(model_hparams=decoder_hparams)

    @property
    def output_size(self) -> int:
        return self.decoder.output_size

    def forward(self,
                batch: Batch,
                max_decoding_length: int=100,
                infer_method: str='greedy',
                sample_top_k: int=0,
                sample_p: float=1.0,
                sample_temperature: float=1.0
                ) -> Dict[str, Tensor]:

        memory_output = self.encoder(batch)

        if 'tgt_ids' in batch.keys():  # Compute loss
            assert 'tgt_lengths' in batch.keys()

            outputs = self.decoder(inputs=batch.tgt_ids[:, :-1],
                                   sequence_length=batch.tgt_lengths - 1,
                                   memory_ids=batch.src_ids,
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
                memory_ids=batch.src_ids,
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
                            eval_fns: List[Any]=[]) -> None:
        self.pg_fns = pg_fns
        self.eval_fns = eval_fns
