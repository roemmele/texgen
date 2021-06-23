from typing import Union
import torch
from texar.torch.modules import (GreedyEmbeddingHelper,
                                 TopKSampleEmbeddingHelper,
                                 TopPSampleEmbeddingHelper)


def get_generation_fn(start_tokens: torch.LongTensor,
                      end_token: Union[int, torch.LongTensor],
                      infer_method: str='greedy',
                      sample_top_k: int=0,
                      sample_p: float=1.0,
                      sample_temperature: float=1.0):
    '''Texar helper function defining generation (sampling) method'''
    assert infer_method in ('greedy', 'sample')
    if infer_method == 'greedy':
        generation_fn = GreedyEmbeddingHelper(start_tokens=start_tokens,
                                              end_token=end_token)
    elif infer_method == 'sample':

        if sample_top_k > 0:
            assert sample_p == 1.0, ("Cannot apply both top-k and top-p sampling." +
                                     "If using top-k sampling (top_k > 0), sample_p must equal 1.0")
            generation_fn = TopKSampleEmbeddingHelper(start_tokens=start_tokens,
                                                      end_token=end_token,
                                                      top_k=sample_top_k,
                                                      softmax_temperature=sample_temperature)
        else:
            generation_fn = TopPSampleEmbeddingHelper(start_tokens=start_tokens,
                                                      end_token=end_token,
                                                      p=sample_p,
                                                      softmax_temperature=sample_temperature)
    return generation_fn
