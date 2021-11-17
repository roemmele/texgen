import numpy
import torch
import spacy
from spacy.tokens import Token
import en_core_web_sm  # spacy english model
import texar.torch as tx
from texar.torch.data import GPT2Tokenizer
from texar.torch.run import *
from collections import defaultdict
from texar.torch.data.data.dataset_utils import Batch
from texar.torch.data.tokenizers.gpt2_tokenizer import GPT2Tokenizer
from typing import Dict, List, Optional, Tuple, Union, Iterator, Set
from spacy.tokens.doc import Doc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

spacy_model = None


def load_tokenizer(tokenizer_name: str) -> GPT2Tokenizer:
    tokenizer = GPT2Tokenizer(tokenizer_name)
    return tokenizer


class TexarDataset(tx.data.DatasetBase):
    '''Texar dataset wrapper required for training API
    process() and collate() functions are required by the library'''

    def __init__(self,
                 tokenizer: GPT2Tokenizer,
                 src_texts_source: Iterator[str],
                 tgt_texts_source: Iterator[str],
                 ref_texts_source: Optional[Iterator[str]] = None,
                 max_src_length: int = 25,
                 max_tgt_length: int = 75,
                 batch_size: int = 1
                 ) -> None:
        self.tokenizer = tokenizer
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.has_added_refs = False
        if ref_texts_source is not None:
            self.has_added_refs = True
            self.data_source = tx.data.ZipDataSource(src_texts_source,
                                                     tgt_texts_source,
                                                     ref_texts_source)
        else:
            self.data_source = tx.data.ZipDataSource(src_texts_source,
                                                     tgt_texts_source)
        super().__init__(source=self.data_source,
                         hparams={'batch_size': batch_size},
                         device=device)

    def process(self,
                raw_example: Tuple[str, str]
                ) -> Dict[str, Union[str, List[int], List[str], int]]:
        if self.has_added_refs:
            src_text, tgt_text, ref_texts = raw_example
        else:
            src_text, tgt_text = raw_example
            ref_texts = [tgt_text]

        return encode_example(self.tokenizer, src_text,
                              tgt_text=tgt_text, ref_texts=ref_texts,
                              max_src_length=self.max_src_length,
                              max_tgt_length=self.max_tgt_length)

    def collate(self,
                examples: List[Dict[str, Union[str, List[int], List[str], int]]]
                ) -> Batch:
        return convert_to_batch(examples)


def create_texar_dataset(tokenizer: GPT2Tokenizer,
                         train_src_texts: Iterator[str],
                         train_tgt_texts: Iterator[str],
                         eval_src_texts: Iterator[str],
                         eval_tgt_texts: Iterator[str],
                         train_ref_texts: Optional[Iterator[str]] = None,
                         eval_ref_texts: Optional[Iterator[str]] = None,
                         max_src_length: int = 25,
                         max_tgt_length: int = 75,
                         batch_size: int = 32
                         ) -> Tuple[TexarDataset, TexarDataset]:

    train_data = TexarDataset(tokenizer=tokenizer,
                              src_texts_source=train_src_texts,
                              tgt_texts_source=train_tgt_texts,
                              ref_texts_source=train_ref_texts,
                              max_src_length=max_src_length,
                              max_tgt_length=max_tgt_length,
                              batch_size=batch_size)
    eval_data = TexarDataset(tokenizer=tokenizer,
                             src_texts_source=eval_src_texts,
                             tgt_texts_source=eval_tgt_texts,
                             ref_texts_source=eval_ref_texts,
                             max_src_length=max_src_length,
                             max_tgt_length=max_tgt_length,
                             batch_size=batch_size)
    return train_data, eval_data


def encode_text(tokenizer: GPT2Tokenizer,
                text: str,
                prepend_token: Optional[str] = None,
                max_seq_length: int = 75,
                include_bos_token: bool = True,
                include_eos_token: bool = True
                ) -> Tuple[List[int], int]:

    if prepend_token:
        text = prepend_token + text

    text_ids, text_length = tokenizer.encode_text(text,
                                                  max_seq_length=max_seq_length,
                                                  append_eos_token=include_eos_token)

    if include_bos_token == False:
        text_ids = text_ids[1:]
        text_length = text_length - 1

    return text_ids, text_length


def encode_into_spacy(text: str,
                      use_profanity_filter: bool = False
                      ) -> Doc:
    global spacy_model
    if spacy_model is None:
        spacy_model = en_core_web_sm.load()
        Token.set_extension('match_text', default="")
    if use_profanity_filter and not spacy_model.has_pipe("profanity_filter"):
        from profanity_filter import ProfanityFilter
        profanity_filter = ProfanityFilter(nlps={'en': spacy_model})
        spacy_model.add_pipe(profanity_filter.spacy_component, last=True)
    spacy_text = spacy_model(text)
    return spacy_text


def get_src_gen_alignment_idxs(spacy_gen_text: Doc,
                               src_tokens: List[str],
                               match_lowercase: bool = True
                               ) -> List[Tuple[int, int]]:
    gen_tokens = [token.text.lower() for token in spacy_gen_text]
    src_token_start_char_idxs = []
    src_token_end_char_idxs = []
    next_token_idx = 0
    for src_token in src_tokens:
        if match_lowercase:
            src_token = src_token.lower()
        src_token_found = False
        for gen_token_idx, gen_token in enumerate(spacy_gen_text[next_token_idx:],
                                                  start=next_token_idx):
            if gen_token.text.lower() == src_token:
                start_char_idx = gen_token.idx
                end_char_idx = start_char_idx + len(src_token)
                src_token_start_char_idxs.append(start_char_idx)
                src_token_end_char_idxs.append(end_char_idx)
                src_token_found = True
                break
        if src_token_found == False:
            break
        next_token_idx = gen_token_idx + 1

    return list(zip(src_token_start_char_idxs,
                    src_token_end_char_idxs))


def get_gen_length_from_ws_split(gen_text: str) -> int:
    '''Length in terms of tokens split on whitespace'''
    return len(gen_text.split(" "))


def get_gen_redundancy_rate(gen_tokens: List[str],
                            input_tokens: List[str],
                            prev_gen_tokens: Set[str]
                            ) -> Tuple[float, Set[str]]:

    noninput_gen_tokens = set(gen_tokens) - set(input_tokens)
    overlap_tokens = prev_gen_tokens.intersection(noninput_gen_tokens)
    if not len(noninput_gen_tokens):
        redund_rate = 1.0
    else:
        redund_rate = len(overlap_tokens) / len(noninput_gen_tokens)
    return redund_rate, overlap_tokens


def get_context_for_regeneration(spacy_src_text: Doc,
                                 spacy_gen_text: Doc
                                 ) -> Union[None, str]:

    last_aligned_gen_idx = -1
    src_last_space = ""
    for tok in spacy_src_text:
        tok._.match_text = tok.text

    for gen_tok_idx, gen_tok in enumerate(spacy_gen_text):

        next_src_tok = spacy_src_text[0]._.match_text.lower()
        gen_tok = gen_tok.lower_

        if (gen_tok.startswith(next_src_tok)
                or gen_tok.endswith(next_src_tok)):
            src_last_space = spacy_src_text[0].whitespace_
            spacy_src_text = spacy_src_text[1:]
            last_aligned_gen_idx = gen_tok_idx
        elif next_src_tok.startswith(gen_tok):
            spacy_src_text[0]._.match_text = spacy_src_text[0]._.match_text[len(
                gen_tok):]

        if not len(spacy_src_text):
            break

    if len(spacy_src_text) and spacy_src_text[0].text.strip():
        first_unaligned_src_tok = spacy_src_text[0].text
        gen_prefix = spacy_gen_text[:last_aligned_gen_idx + 1].text_with_ws
        if gen_prefix and gen_prefix[-1] != " " and src_last_space == " ":
            gen_prefix += src_last_space
        regen_ctx_text = gen_prefix + first_unaligned_src_tok
        return regen_ctx_text

    return None


def encode_example(tokenizer: GPT2Tokenizer,
                   src_text: str,
                   tgt_text: Optional[str] = None,
                   ctx_text: Optional[str] = None,
                   ref_texts: Optional[List[str]] = None,
                   src_prepend_token: str = '{{',
                   tgt_prepend_token: str = '}}',
                   max_src_length: int = 25,
                   max_tgt_length: int = 75
                   ) -> Dict[str, Union[str, List[int], List[str], int]]:

    src_ids, src_length = encode_text(tokenizer, src_text,
                                      prepend_token=src_prepend_token,
                                      max_seq_length=max_src_length,
                                      include_eos_token=False)
    enc_example = {'src_text': src_text,
                   'src_ids': src_ids[:src_length]}

    if ctx_text is not None:
        ctx_ids, ctx_length = encode_text(tokenizer, ctx_text,
                                          prepend_token=tgt_prepend_token,
                                          max_seq_length=max_tgt_length,
                                          include_bos_token=False,
                                          include_eos_token=False)
        enc_example.update({'ctx_text': ctx_text,
                            'ctx_ids': ctx_ids[:ctx_length]})

    if tgt_text is not None:
        tgt_ids, tgt_length = encode_text(tokenizer, tgt_text,
                                          prepend_token=tgt_prepend_token,
                                          include_bos_token=False,
                                          max_seq_length=max_tgt_length)
        enc_example.update({'tgt_text': tgt_text,
                            'tgt_ids': tgt_ids[:tgt_length]})

        if ref_texts is not None:
            enc_example['ref_texts'] = ref_texts

    enc_example.update({'bos_token_id': tokenizer.map_token_to_id(tokenizer.bos_token),
                        'eos_token_id': tokenizer.map_token_to_id(tokenizer.eos_token),
                        'pad_token_id': tokenizer.map_token_to_id(tokenizer.pad_token)})

    return enc_example


def convert_to_batch(examples: Union[List[Dict[str, Union[str, List[int], int]]],
                                     List[Dict[str, Union[str, List[int], List[str], int]]]]
                     ) -> Batch:

    pad_token_id = examples[0]['pad_token_id']
    batch = {}

    src_ids, src_lengths = tx.data.padded_batch([example["src_ids"] for example in examples],
                                                pad_value=pad_token_id)
    batch.update({'src_ids': torch.from_numpy(src_ids).to(device),
                  'src_lengths': torch.tensor(src_lengths).to(device)})

    if 'ctx_ids' in examples[0]:
        ctx_ids, ctx_lengths = tx.data.padded_batch([example["ctx_ids"] for example in examples],
                                                    pad_value=pad_token_id)
        batch.update({'ctx_ids': torch.from_numpy(ctx_ids).to(device),
                      'ctx_lengths': torch.tensor(ctx_lengths).to(device)})

    if 'tgt_ids' in examples[0]:
        tgt_ids, tgt_lengths = tx.data.padded_batch([example["tgt_ids"] for example in examples],
                                                    pad_value=pad_token_id)
        batch.update({'tgt_ids': torch.from_numpy(tgt_ids).to(device),
                      'tgt_lengths': torch.tensor(tgt_lengths).to(device)})

        if 'ref_texts' in examples[0]:
            batch['ref_texts'] = [example['ref_texts'] for example in examples]

    # Infer pad, start, and end tokens from first example in batch
    batch.update({'bos_token_id': examples[0]['bos_token_id'],
                  'eos_token_id': examples[0]['eos_token_id'],
                  'pad_token_id': pad_token_id})

    batch = tx.data.Batch(batch_size=len(examples), **batch)

    return batch
