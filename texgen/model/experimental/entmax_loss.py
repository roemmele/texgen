from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from texar.torch.losses.losses_utils import mask_and_reduce, reduce_dimensions
from texar.torch.utils import shapes
from texar.torch.utils.types import MaybeTuple
from entmax import (SparsemaxLoss, Entmax15Loss, EntmaxBisectLoss,
                    sparsemax, entmax15, entmax_bisect, entmax_bisect_loss)
from functools import partial

# entmax_loss = partial(
#     EntmaxBisectLoss,
#     alpha=1.2,
#     n_iter=50
# )

# entmax = partial(
#     entmax_bisect,
#     alpha=1.2,
#     n_iter=50
# )


def sequence_sparse_entmax_cross_entropy(
        labels: torch.Tensor,
        logits: torch.Tensor,
        sequence_length: Optional[torch.LongTensor],
        average_across_batch: bool = True,
        average_across_timesteps: bool = False,
        sum_over_batch: bool = False,
        sum_over_timesteps: bool = True,
        time_major: bool = False) -> torch.Tensor:
    r"""Computes sparse softmax cross entropy for each time step of sequence
    predictions.
    Args:
        labels: Target class indexes. I.e., classes are mutually exclusive
            (each entry is in exactly one class).
            - If :attr:`time_major` is `False` (default), this must be
              a Tensor of shape `[batch_size, max_time]`.
            - If `time_major` is `True`, this must be a Tensor of shape
              `[max_time, batch_size].`
        logits: Unscaled log probabilities. This must have the shape of
            `[max_time, batch_size, num_classes]` or
            `[batch_size, max_time, num_classes]` according to
            the value of `time_major`.
        sequence_length: A Tensor of shape `[batch_size]`. Time steps beyond
            the respective sequence lengths will have zero losses.
        average_across_timesteps (bool): If set, average the loss across
            the time dimension. Must not set `average_across_timesteps`
            and `sum_over_timesteps` at the same time.
        average_across_batch (bool): If set, average the loss across the
            batch dimension. Must not set `average_across_batch`'
            and `sum_over_batch` at the same time.
        sum_over_timesteps (bool): If set, sum the loss across the
            time dimension. Must not set `average_across_timesteps`
            and `sum_over_timesteps` at the same time.
        sum_over_batch (bool): If set, sum the loss across the
            batch dimension. Must not set `average_across_batch`
            and `sum_over_batch` at the same time.
        time_major (bool): The shape format of the inputs. If `True`,
            :attr:`labels` and :attr:`logits` must have shape
            `[max_time, batch_size, ...]`. If `False`
            (default), they must have shape `[batch_size, max_time, ...]`.
    Returns:
        A Tensor containing the loss, of rank 0, 1, or 2 depending on the
        arguments :attr:`{average_across}/{sum_over}_{timesteps}/{batch}`.
        For example:
        - If :attr:`sum_over_timesteps` and :attr:`average_across_batch`
          are `True` (default), the return Tensor is of rank 0.
        - If :attr:`average_across_batch` is `True` and other arguments are
          `False`, the return Tensor is of shape `[max_time]`.
    Example:
        .. code-block:: python
            embedder = WordEmbedder(vocab_size=data.vocab.size)
            decoder = BasicRNNDecoder(vocab_size=data.vocab.size)
            outputs, _, _ = decoder(
                decoding_strategy='train_greedy',
                inputs=embedder(data_batch['text_ids']),
                sequence_length=data_batch['length']-1)
            loss = sequence_sparse_softmax_cross_entropy(
                labels=data_batch['text_ids'][:, 1:],
                logits=outputs.logits,
                sequence_length=data_batch['length']-1)
    """
    # import pdb
    # pdb.set_trace()
    #logits = F.log_softmax(logits, dim=2)
    logits = entmax_bisect(logits, dim=2)
    #logits = logits.permute(0, 2, 1)
    #losses = F.nll_loss(logits, labels, reduction='none')
    losses = entmax_bisect_loss(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                                alpha=1.2, n_iter=50)
    losses = losses.reshape(-1, labels.shape[-1])
    #losses = entmax_bisect_loss(logits, labels, alpha=1.2, n_iter=50)

    losses = mask_and_reduce(losses,
                             sequence_length,
                             rank=2,
                             average_across_batch=average_across_batch,
                             average_across_timesteps=average_across_timesteps,
                             sum_over_batch=sum_over_batch,
                             sum_over_timesteps=sum_over_timesteps,
                             time_major=time_major)
    return losses
