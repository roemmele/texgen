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
                 embedder_hparams: Optional[Dict[str, Any]]=None,
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
            self.encoder = TransformerEncoderWrapper(hparams=encoder_hparams)

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
            self.decoder = TransformerDecoderWrapper(hparams=decoder_hparams)

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

    # def compute_mle_loss(self, batch, enc_states):

    #     mle_outputs = self.decoder(batch, enc_states, return_mle=True)
    #     mle_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
    #         labels=batch.tgt_ids[:, 1:],
    #         logits=mle_outputs.logits,
    #         sequence_length=batch.tgt_lengths - 1
    #     )

    #     return mle_loss

    # def compute_pg_losses(self, batch, enc_states):

    #     (argmax_outputs,
    #      argmax_lengths) = self.decoder(batch,
    #                                     enc_states)
    #     # infer_method='greedy')

    #     argmax_scores = self.compute_eval_scores(eval_fns=self.pg_fns,
    #                                              batch=batch,
    #                                              outputs=argmax_outputs,
    #                                              output_lengths=argmax_lengths)

    #     if device.type == 'cuda':  # Free some memory
    #         del argmax_outputs
    #         del argmax_lengths
    #         torch.cuda.empty_cache()

    #     (rand_sample_outputs,
    #      rand_sample_lengths) = self.decoder(batch,
    #                                          enc_states,
    #                                          decoding_strategy='infer_sample')

    #     rand_sample_log_probs = rand_sample_outputs.logits.log_softmax(-1)[
    #         torch.arange(len(batch)).unsqueeze(-1),
    #         torch.arange(rand_sample_outputs.sample_id.shape[-1]),
    #         rand_sample_outputs.sample_id]

    #     rand_sample_scores = self.compute_eval_scores(eval_fns=self.pg_fns,
    #                                                   batch=batch,
    #                                                   outputs=rand_sample_outputs,
    #                                                   output_lengths=rand_sample_lengths)

    #     pg_losses = {}
    #     for pg_fn in self.pg_fns:
    #         advantages = (rand_sample_scores[pg_fn.metric_name]
    #                       - argmax_scores[pg_fn.metric_name])
    #         pg_loss = tx.losses.pg_loss_with_log_probs(
    #             log_probs=rand_sample_log_probs,
    #             advantages=advantages.unsqueeze(-1).expand(-1, rand_sample_log_probs.shape[-1]),
    #             batched=True,
    #             sequence_length=rand_sample_lengths,
    #             sum_over_timesteps=True)
    #         pg_losses["{}_loss".format(pg_fn.metric_name)] = pg_loss

    #     return pg_losses

    # def compute_eval_scores(self, eval_fns, batch, outputs,
    #                         output_lengths, reduce_mean=False):
    #     eval_scores = {}
    #     for eval_fn in eval_fns:
    #         if eval_fn.requires_src and eval_fn.requires_refs:
    #             scores = [eval_fn(src=src_ids[:src_length].cpu().numpy(),
    #                               out=pred_ids[:pred_length].cpu().numpy(),
    #                               refs=ref_texts,
    #                               decode=True)
    #                       for src_ids, src_length, pred_ids, pred_length, ref_texts
    #                       in zip(batch.src_ids, batch.src_lengths,
    #                              outputs.sample_id, output_lengths, batch.ref_texts)]
    #         elif eval_fn.requires_src:
    #             scores = [eval_fn(src=src_ids[:src_length].cpu().numpy(),
    #                               out=pred_ids[:pred_length].cpu().numpy(),
    #                               decode=True)
    #                       for src_ids, src_length, pred_ids, pred_length
    #                       in zip(batch.src_ids, batch.src_lengths,
    #                              outputs.sample_id, output_lengths)]
    #         elif eval_fn.requires_refs:
    #             scores = [eval_fn(out=pred_ids[:pred_length].cpu().numpy(),
    #                               refs=ref_texts,
    #                               decode=True)
    #                       for pred_ids, pred_length, ref_texts
    #                       in zip(outputs.sample_id, output_lengths, batch.ref_texts)]

    #         else:
    #             scores = [eval_fn(out=pred_ids[:pred_length].cpu().numpy(),
    #                               decode=True)
    #                       for pred_ids, pred_length
    #                       in zip(outputs.sample_id, output_lengths)]

    #         eval_scores[eval_fn.metric_name] = torch.from_numpy(numpy.array(scores,
    #                                                                         dtype='float32')).to(device)
    #         if reduce_mean:
    #             eval_scores[eval_fn.metric_name] = torch.mean(eval_scores[eval_fn.metric_name])

    #     return eval_scores
