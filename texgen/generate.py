import warnings
warnings.filterwarnings('ignore')

from typing import Set, List, Optional, Iterator
import logging
from collections import Counter
from argparse import Namespace
from spacy.tokens.doc import Doc

import torch
from texar.torch.data.tokenizers.gpt2_tokenizer import GPT2Tokenizer

from .data import (encode_example, convert_to_batch, load_tokenizer,
                   encode_into_spacy, get_context_for_regeneration,
                   get_gen_redundancy_rate, get_gen_length_from_ws_split,
                   get_src_gen_alignment_idxs)
from .construct import load_model
from .args import get_generation_parser
from .model.lm import LM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)

logger = logging.getLogger(__name__)


def generate_batch(batch_src_texts: List[str],
                   tokenizer: GPT2Tokenizer,
                   model: LM,
                   max_decoding_length: int,
                   batch_ctx_texts: Optional[List[str]]=None,
                   infer_method: str='greedy',
                   sample_top_k: int=0,
                   sample_p: float=1.0,
                   sample_temperature: float=1.0
                   ) -> List[str]:

    if batch_ctx_texts:
        encoded_examples = [encode_example(tokenizer, src_text, ctx_text=ctx_text)
                            for src_text, ctx_text in zip(batch_src_texts, batch_ctx_texts)]
    else:
        encoded_examples = [encode_example(tokenizer, src_text, ctx_text="")
                            for src_text in batch_src_texts]

    batched_examples = convert_to_batch(encoded_examples)

    with torch.no_grad():
        pred_output = model(batch=batched_examples,
                            max_decoding_length=max_decoding_length,
                            infer_method=infer_method,
                            sample_top_k=sample_top_k,
                            sample_p=sample_p,
                            sample_temperature=sample_temperature)

    # Omit prepended start token when returning generated text
    batch_gen_texts = [tokenizer.map_id_to_text(pred_ids[1:pred_length].cpu().numpy(),
                                                skip_special_tokens=True).strip()
                       for pred_ids, pred_length in zip(pred_output['pred_ids'],
                                                        pred_output['pred_lengths'])]

    return batch_gen_texts


def check_if_needs_regen(spacy_gen_text: Doc,
                         src_tokens: Optional[List[str]]=None,
                         min_postproc_length: Optional[int]=None,
                         max_postproc_length: Optional[int]=None,
                         require_src_in_gen: bool=False,
                         block_repeat: bool=False,
                         block_quotes: bool=False,
                         block_profanity: bool=False,
                         require_eos_punct: bool=False,
                         require_paired_punct: bool=False,
                         max_redundancy_rate: Optional[float]=None,
                         n_gen_per_src: int=1,
                         prev_gen_tokens: Optional[Set[str]]=None,
                         verbose: bool=False
                         ) -> bool:

    gen_length = get_gen_length_from_ws_split(spacy_gen_text.text)
    if ((min_postproc_length and gen_length < min_postproc_length)
            or (max_postproc_length and gen_length > max_postproc_length)):
        if verbose:
            logger.info("Failed with length = {} (required {}<=length<={}): {}".format(
                gen_length, min_postproc_length, max_postproc_length, spacy_gen_text.text))
        return True

    if block_quotes:
        if '"' in set(spacy_gen_text.text):
            if verbose:
                logger.info("Failed with quotation mark: {}".format(spacy_gen_text.text))
            return True

    if require_eos_punct:
        if spacy_gen_text.text[-1].isalnum():
            if verbose:
                logger.info("Failed with no end-of-sentence punctuation: {}".format(spacy_gen_text.text))
            return True

    if block_repeat:
        src_token_counts = Counter(src_tokens)
        for token, next_token in zip(spacy_gen_text[:-1], spacy_gen_text[1:]):
            if token.text.lower() == next_token.text.lower():
                if token.text.lower() not in src_token_counts or src_token_counts[token.text.lower()] == 1:
                    if verbose:
                        logger.info("Failed with repeated token = {}: {}".format(token.text, spacy_gen_text.text))
                    return True
                elif token.text.lower() in src_token_counts:
                    src_token_counts[token.text.lower()] -= 1

    if require_paired_punct:
        chars = Counter(spacy_gen_text.text)
        if ((chars['"'] % 2 != 0) or
                ("(" in chars and ")" not in chars) or (")" in chars and "(" not in chars) or
                ("[" in chars and "]" not in chars) or ("]" in chars and "[" not in chars) or
                ("{" in chars and "}" not in chars) or ("}" in chars and "{" not in chars)):
            if verbose:
                logger.info("Failed with unpaired punctuation: {}".format(spacy_gen_text.text))
            return True

    if require_src_in_gen:
        src_token_idxs = get_src_gen_alignment_idxs(spacy_gen_text,
                                                    src_tokens)
        if len(src_token_idxs) < len(src_tokens):
            if verbose:
                logger.info("Failed with missing source token (required tokens = {}): {}".format(src_tokens, spacy_gen_text.text))
            return True

    if max_redundancy_rate and n_gen_per_src > 1:
        gen_tokens = [token.text.lower() for token in spacy_gen_text]
        redund_rate, overlap_tokens = get_gen_redundancy_rate(gen_tokens=gen_tokens,
                                                              input_tokens=src_tokens,
                                                              prev_gen_tokens=prev_gen_tokens)
        if redund_rate > max_redundancy_rate:
            if verbose:
                logger.info("Failed with redund_rate>{} (score = {:.3f}, overlapping tokens = {}): {}".format(
                    max_redundancy_rate, redund_rate, overlap_tokens, spacy_gen_text.text)
                )
            return True

    if block_profanity:
        if spacy_gen_text._.is_profane:
            logger.info("Failed with profanity: {}".format(spacy_gen_text.text))
            return True

    return False


def generate(args: Namespace,
             src_texts: Iterator[str]
             ) -> None:

    model, model_hparams = load_model(load_dir=args.model_dir)
    model.eval()

    tokenizer = load_tokenizer(model_hparams['tokenizer'])

    immut_src_texts = list(src_texts)

    regen_is_possible = (args.require_src_in_gen
                         or args.min_postproc_length != None
                         or args.max_postproc_length != None
                         or (args.max_redundancy_rate != None and args.n_gen_per_src > 1))

    if regen_is_possible:
        immut_spacy_src_texts = [encode_into_spacy(text) for text in immut_src_texts]
        if args.max_redundancy_rate and args.n_gen_per_src > 1:
            gen_token_sets = [set() for _ in range(len(immut_src_texts))]

    gen_texts_by_round = []

    for gen_round in range(args.n_gen_per_src):

        src_texts = immut_src_texts
        src_order_idxs = list(range(len(src_texts)))
        gen_texts = [None for _ in src_order_idxs]
        if regen_is_possible:
            spacy_src_texts = immut_spacy_src_texts
            if args.force_src_in_regen:
                ctx_texts = ["" for _ in src_order_idxs]

        gen_attempt = 1

        while gen_attempt <= args.max_gen_attempts:

            logger.info("Starting generation round {}, attempt {}...".format(gen_round + 1, gen_attempt))

            unfinished_gen_idxs = []

            for batch_idx in range(0, len(src_texts), args.batch_size):
                batch_src_order_idxs = src_order_idxs[batch_idx:batch_idx + args.batch_size]
                batch_src_texts = src_texts[batch_idx:batch_idx + args.batch_size]

                if not (args.require_src_in_gen and args.force_src_in_regen) or gen_attempt == 1:
                    batch_ctx_texts = None
                else:
                    batch_ctx_texts = ctx_texts[batch_idx:batch_idx + args.batch_size]

                batch_gen_texts = generate_batch(batch_src_texts=batch_src_texts,
                                                 tokenizer=tokenizer,
                                                 model=model,
                                                 max_decoding_length=args.max_decoding_length,
                                                 batch_ctx_texts=batch_ctx_texts,
                                                 infer_method=args.infer_method,
                                                 sample_top_k=args.sample_top_k,
                                                 sample_p=args.sample_p,
                                                 sample_temperature=args.sample_temperature)

                if regen_is_possible:
                    batch_spacy_src_texts = spacy_src_texts[batch_idx:batch_idx + args.batch_size]
                else:
                    batch_spacy_src_texts = [None] * args.batch_size

                for (item_idx, (src_order_idx, spacy_src_text, gen_text))\
                    in enumerate(zip(batch_src_order_idxs,
                                     batch_spacy_src_texts,
                                     batch_gen_texts), start=batch_idx):

                    if regen_is_possible:
                        spacy_gen_text = encode_into_spacy(gen_text,
                                                           use_profanity_filter=args.block_profanity)
                        src_tokens = [token.text for token in spacy_src_text]

                        needs_regen = check_if_needs_regen(
                            spacy_gen_text,
                            src_tokens=src_tokens,
                            min_postproc_length=args.min_postproc_length,
                            max_postproc_length=args.max_postproc_length,
                            require_src_in_gen=args.require_src_in_gen,
                            block_repeat=args.block_repeat,
                            block_quotes=args.block_quotes,
                            block_profanity=args.block_profanity,
                            require_eos_punct=args.require_eos_punct,
                            require_paired_punct=args.require_paired_punct,
                            max_redundancy_rate=args.max_redundancy_rate,
                            n_gen_per_src=args.n_gen_per_src,
                            prev_gen_tokens=(gen_token_sets[src_order_idx]
                                             if (args.max_redundancy_rate and args.n_gen_per_src > 1) else None),
                            verbose=args.verbose
                        )

                        if needs_regen:
                            unfinished_gen_idxs.append(item_idx)
                            if gen_attempt < args.max_gen_attempts:
                                if args.force_src_in_regen:
                                    regen_ctx_text = get_context_for_regeneration(
                                        spacy_src_text,
                                        encode_into_spacy(gen_text)
                                    )
                                    ctx_texts[item_idx] = regen_ctx_text
                                continue
                            else:
                                if args.fallback_to_src:
                                    gen_text = src_text
                                else:
                                    gen_text = ""

                        if args.max_redundancy_rate and args.n_gen_per_src > 1:
                            gen_tokens = [token.text.lower() for token in spacy_gen_text]
                            gen_token_sets[src_order_idx].update(set(gen_tokens) - set(src_tokens))

                    gen_texts[src_order_idx] = gen_text

                if (batch_idx % (args.batch_size * 5) == 0
                        or (batch_idx + args.batch_size) >= len(src_texts)):
                    logger.info("round {}, attempt {}: generated {} texts ({} failed requirements)...".format(
                        gen_round + 1,
                        gen_attempt,
                        min(len(src_texts), batch_idx + args.batch_size),
                        len(unfinished_gen_idxs))
                    )

            gen_attempt += 1

            if not unfinished_gen_idxs:
                break

            src_order_idxs = [src_order_idxs[idx] for idx in unfinished_gen_idxs]
            src_texts = [src_texts[idx] for idx in unfinished_gen_idxs]

            if regen_is_possible:
                spacy_src_texts = [spacy_src_texts[idx] for idx in unfinished_gen_idxs]
                if args.force_src_in_regen:
                    ctx_texts = [ctx_texts[idx] for idx in unfinished_gen_idxs]

        gen_texts_by_round.append(gen_texts)

    final_gen_texts = list(zip(*gen_texts_by_round))

    if args.gen_texts_file is not None:
        with open(args.gen_texts_file, 'w') as f:
            f.write("\n".join(["\t".join(item_gen_texts)
                               for item_gen_texts in final_gen_texts]))
        logger.info("Saved generated texts to {}".format(args.gen_texts_file))

    return final_gen_texts
