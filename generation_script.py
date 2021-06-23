import os
import logging
from texgen.generate import generate
from texgen.args import get_generation_parser

logging.basicConfig(level=os.environ.get('LOGGING_LEVEL', 'INFO'))

if __name__ == "__main__":
    parser = get_generation_parser()
    args = parser.parse_args()
    print(vars(args))

    with open(args.src_texts_file) as f:
        src_texts = [text.strip() for text in f]

    generate(args=args, src_texts=src_texts)
