import os
import logging
from texgen.train import train
from texgen.args import get_train_parser

logging.basicConfig(level=os.environ.get('LOGGING_LEVEL', 'INFO'))

if __name__ == '__main__':
    parser = get_train_parser()
    args = parser.parse_args()
    print(vars(args))

    # Load train data
    with open(args.train_src_file) as f:
        train_src_texts = [text.strip() for text in f]
    with open(args.train_tgt_file) as f:
        train_tgt_texts = [text.strip() for text in f]

    # Load validation data
    with open(args.eval_src_file) as f:
        eval_src_texts = [text.strip() for text in f]
    with open(args.eval_tgt_file) as f:
        eval_tgt_texts = [text.strip() for text in f]

    # Optional: Load additional targets (references)
    if args.train_ref_file is not None and args.eval_ref_file is not None:
        with open(args.train_ref_file) as f:
            train_ref_texts = [text.strip() for text in f]
        with open(args.eval_ref_file) as f:
            eval_ref_texts = [text.strip() for text in f]
    else:
        train_ref_texts, eval_ref_texts = None, None

    train(args=args,
          train_src_texts=train_src_texts,
          train_tgt_texts=train_tgt_texts,
          eval_src_texts=eval_src_texts,
          eval_tgt_texts=eval_tgt_texts,
          train_ref_texts=train_ref_texts,
          eval_ref_texts=eval_ref_texts)
