import argparse
from .metric import RawMetric
from argparse import ArgumentParser


def get_argparser(description: str) -> ArgumentParser:
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    return parser


def get_train_parser() -> ArgumentParser:
    parser = get_argparser(description=("Train an encoder-decoder model (i.e. seq2seq) " +
                                        "or decoder-only model (i.e. language model)"))
    add_train_args(parser)
    return parser


def get_generation_parser() -> ArgumentParser:
    parser = get_argparser(description=("Generate output texts from input source texts " +
                                        "given trained model."))
    add_generation_args(parser)
    return parser


def add_train_args(parser: ArgumentParser) -> None:
    parser.add_argument("--save_dir", "-save_dir", help="Provide the directory path where the model checkpoint files will be saved.",
                        type=str, required=True)
    parser.add_argument("--config_file", "-config_file", help="Specify path to .json config file where model definition/hyperparameters are given.\
                        If loading an existing model from args.load_from_dir, this file will be ignored and the saved config for the loaded model be read.",
                        type=str, required=False, default=None)

    parser.add_argument("--train_src_file", "-train_src_file", help="Specify path to .txt file containing train source texts (one text per line)",
                        type=str, required=False, default=None)
    parser.add_argument("--train_tgt_file", "-train_tgt_file", help="Specify path to .txt file containing train target texts (one text per line)",
                        type=str, required=False, default=None)
    parser.add_argument("--eval_src_file", "-eval_src_file", help="Specify path to .txt file containing source texts used for evaluation (one text per line)",
                        type=str, required=False, default=None)
    parser.add_argument("--eval_tgt_file", "-eval_tgt_file", help="Specify path to .txt file containing target texts used for evaluation (one text per line)",
                        type=str, required=False, default=None)
    parser.add_argument("--train_ref_file", "-train_ref_file", help="Specify path to .txt file containing additional reference target texts used for training (separate multiple references per source with tabs).",
                        type=str, required=False, default=None)
    parser.add_argument("--eval_ref_file", "-eval_ref_file", help="Specify path to .txt file containing additional reference target texts used for evaluation (separate multiple references per source with tabs).",
                        type=str, required=False, default=None)

    parser.add_argument("--max_src_length", "-max_src_length", help="Specify maximum number of tokens in source texts. Texts longer than this will be truncated.",
                        type=int, required=False, default=25)
    parser.add_argument("--max_tgt_length", "-max_tgt_length", help="Specify maximum number of tokens in target texts. Texts longer than this will be truncated.",
                        type=int, required=False, default=75)
    parser.add_argument("--load_from_dir", "-load_from_dir", help="Provide directory path of existing model checkpoints used to initialize weights of new model.\
                        If multiple checkpoints exist in this directory, will load the one with the most recent timestep.",
                        type=str, required=False, default=None)
    parser.add_argument("--pg_metrics", "-pg_metrics", help="Provide names of policy gradient metrics used for reinforcement learning (separate metric names with space).",
                        type=str, nargs='+', choices=list(RawMetric.get_metric_names().keys()), required=False, default=[])
    parser.add_argument("--eval_metrics", "-eval_metrics", help="Provide names of metrics applied during validation (separate metric names with space).\
                        You should list the metrics in order of \"importance\" for early stopping criteria; early stopping will mark improvements in the first metric;\
                        if those are equal, it looks at improvements in the second metric, etc.\
                        By default MLE loss will also be computed as the first metric (so it is the first consideration for early stopping).",
                        type=str, nargs='+', required=False, choices=list(RawMetric.get_metric_names().keys()), default=[])
    parser.add_argument("--batch_size", "-batch_size", help="Specify training batch size.",
                        type=int, required=False, default=32)
    parser.add_argument("--max_epochs", "-max_epochs", help="Specify maximum number of iterations through dataset (epochs) to train for.",
                        type=int, required=False, default=100)
    parser.add_argument("--learning_rate", "-learning_rate", help="Specify learning rate for training model.",
                        type=float, required=False, default=1e-3)
    parser.add_argument("--patience", "-patience", help="Specify how many epochs to wait without improvement in validation metrics before terminating training.",
                        type=int, required=False, default=2)
    parser.add_argument("--dynamic_lr", "-dynamic_lr", help="As alternative to specifying static learning rate, set learning rate dynamically according to training steps,\
                        using the Noam method. This ignores static learning rate parameter.",
                        action='store_true', required=False)
    parser.add_argument("--warmup_steps", "-warmup_steps", help="Only applicable is using dynamic learning rate.\
                        Specify number of iterations at beginning of training during which learning rate will gradually be increased, after which learning rate will decay.",
                        type=int, required=False, default=4000)
    parser.add_argument("--max_grad_norm", "-max_grad_norm", help="Apply gradient clipping if norm of gradient exceeds specified value.",
                        type=float, required=False, default=5.0)
    parser.add_argument("--accum_steps", "-accum_steps", help="Specify number of steps in which to accumulate gradients (simulates having larger batch size).",
                        type=int, required=False, default=1)
    parser.add_argument("--log_iterations", "-log_iterations", help="Specify how often to report training metrics (iteration=batch)",
                        type=int, required=False, default=100)
    parser.add_argument("--valid_iterations", "-valid_iterations", help="Specify how often to run validation metrics and save improved model (iteration=batch)",
                        type=int, required=False, default=1000)
    parser.add_argument("--valid_epoch_end", "-valid_epoch_end", help="Run validation metrics at the end of every epoch in addition to every N=valid_iterations",
                        action='store_true', required=False)


def add_generation_args(parser: ArgumentParser) -> None:
    parser.add_argument("--model_dir", "-model_dir", help="Specify directory path of trained model.\
                        This directory should include the model hyperparameters (.json file) and the model checkpoint (.pt file)",
                        type=str, required=True)

    parser.add_argument("--gen_texts_file", "-gen_texts_file", help="Specify filepath to which generated output will be saved.",
                        type=str, required=False, default=None)
    parser.add_argument("--src_texts_file", "-src_texts_file", help="Specify path to .txt file containing input texts.",
                        type=str, required=False, default=None)

    parser.add_argument("--max_decoding_length", "-max_decoding_length", help="Specify maximum number of tokens to generate for each input.",
                        type=int, required=False, default=100)
    parser.add_argument("--n_gen_per_src", "-n_gen_per_src", help="Specify number of texts to generate for each source text. Texts will be tab-separated on same line.",
                        type=int, required=False, default=1)
    parser.add_argument("--batch_size", "-batch_size", help="Specify generation batch size.",
                        type=int, required=False, default=64)

    parser.add_argument("--min_postproc_length", "-min_postproc_length", help="Specify minimum number of tokens in generated texts,\
                        such that texts shorter than this will trigger a regeneration attempt. Note that token counting here is just done by splitting on whitespace.",
                        type=int, required=False, default=None)
    parser.add_argument("--max_postproc_length", "-max_postproc_length", help="Specify maximum number of tokens in generated texts,\
                        such that texts longer than this will trigger a regeneration attempt. Note this is different from the max_decoding_length parameter, which is applied during generation instead of as a postprocessing step.\
                        Also note that token counting here is just done by splitting on whitespace.",
                        type=int, required=False, default=None)
    parser.add_argument("--max_redundancy_rate", "-max_redundancy_rate", help="Specify max proportion of tokens in a generated text that can overlap with tokens in other generated texts for the same source.\
                        If this threshold is exceeded (e.g. more than N=redundancy_filter_score proportion of non-input tokens in a generated text already appear in at least one other generated text),\
                        the text is considered redundant and regeneration will trigger. This promotes diversity among the generated texts for a given source.",
                        type=float, required=False, default=None)
    parser.add_argument("--block_repeat", "-block_repeat", help="Trigger re-generation for texts where the same token appears two or more times adjacently, unless token also appears more than once in the source.",
                        action='store_true', required=False)
    parser.add_argument("--block_quotes", "-block_quotes", help="Trigger re-generation for texts that contain quotation marks (\" or \').",
                        action='store_true', required=False)
    parser.add_argument("--block_profanity", "-block_profanity", help="Trigger re-generation for texts that contain profanity (detected by the profanity-filter spacy extension library).",
                        action='store_true', required=False)
    parser.add_argument("--require_paired_punct", "-require_paired_punct", help="Trigger re-generation for texts containing an odd number of quotation marks, or unpaired parentheses/brackets.",
                        action='store_true', required=False)
    parser.add_argument("--require_eos_punct", "-require_eos_punct", help='Trigger re-generation for texts whose last character is alphanumeric (i.e. text is missing end-of-sentence punctuation).',
                        action='store_true', required=False)

    parser.add_argument("--require_src_in_gen", "-require_src_in_gen", help="Require that generated output contain all tokens that appear in source text.",
                        action='store_true', required=False)
    parser.add_argument("--force_src_in_regen", "-force_src_in_regen", help="If require_src_in_gen=True, in cases where generated text does not contain all source tokens,\
                        extract slice of generated text up to last satisfied source token. Append the next unsatisfied source token to this slice, then use this text\
                        as the context for re-generation, such that all subsequently generated tokens are appended to this previously generated segment.\
                        This is most appropriate for source inputs that are assumed to already be a grammatical sentence.\
                        When this parameter is not specified, the model will just re-generate a new text from scratch without any provided context.\
                        This is fitting when there is no assumption that the source text is already a grammatical sentence.",
                        action='store_true', required=False)
    parser.add_argument("--max_gen_attempts", "-max_gen_attempts", help="If require_src_in_gen=True, specify the maximum number of rounds of generation to\
                        try, checking that all source tokens appear in the generated output after that round. If after max_gen_attempts,\
                        the alignment for a given input is still not fulfilled, the generated output for that item will either be an empty string or a copy of the source if fallback_to_src=True.",
                        type=int, required=False, default=1)
    parser.add_argument("--fallback_to_src", "-fallback_to_src", help="If require_src_in_gen=True, and a generated text does not contain all source tokens even after max_gen_attempts,\
                        the generated text will just be a copy of the source text. If require_src_in_gen=True and this parameter is not specified, the generated text is an empty string.",
                        action='store_true', required=False)

    parser.add_argument("--infer_method", "-infer_method", help="Specify inference (decoding) method to use: greedy or sample.",
                        type=str, required=False, choices=['greedy', 'sample'], default='greedy')
    parser.add_argument("--sample_top_k", "-sample_top_k", help="If using sampling inference (i.e. infer_method=sample), specify top-k most probable tokens to sample from.",
                        type=int, required=False, default=0)
    parser.add_argument("--sample_p", "-sample_p", help="If using sampling inference (i.e. infer_method=sample), specify token probability threshold for sampling \
                        (only tokens with probabilities above this threshold will be sampled).",
                        type=float, required=False, default=1.0)
    parser.add_argument("--sample_temperature", "-sample_temperature", help="If using sampling inference, specify softmax temperature parameter.",
                        type=float, required=False, default=1.0)

    parser.add_argument("--verbose", "-verbose", help="Print log of all texts that trigger regeneration.",
                        action='store_true', required=False)
