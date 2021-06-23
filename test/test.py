import os
import shutil
import unittest
import logging

logging.basicConfig(level=os.environ.get('LOGGING_LEVEL', 'INFO'))

from texgen.args import get_train_parser, get_generation_parser
from texgen import train
from texgen import generate


class BaseTestTrain():

    config_filename = None

    def test(self):
        print("Test training model with config:", self.config_filename, "\n")
        parser = get_train_parser()
        args = parser.parse_args(['-save_dir', './test/test_model',
                                  '-config_file', os.path.join('./test/test_configs', self.config_filename),
                                  '-max_epochs', '1',
                                  '-batch_size', '2',
                                  '-max_grad_norm', '1.0',
                                  '-valid_epoch_end'])

        with open("./test/toy_data/train_src.txt") as f:
            train_src_texts = [text.strip() for text in f]
        with open("./test/toy_data/train_tgt.txt") as f:
            train_tgt_texts = [text.strip() for text in f]

        with open("./test/toy_data/eval_src.txt") as f:
            eval_src_texts = [text.strip() for text in f]
        with open("./test/toy_data/eval_tgt.txt") as f:
            eval_tgt_texts = [text.strip() for text in f]

        train(args=args,
              train_src_texts=train_src_texts,
              train_tgt_texts=train_tgt_texts,
              eval_src_texts=train_src_texts,
              eval_tgt_texts=train_tgt_texts)


class BaseTestGenerate():

    def test_default(self):
        print("Test generating from model with config:", self.config_filename, "\n")
        parser = get_generation_parser()
        args = parser.parse_args(['-model_dir', './test/test_model',
                                  '-gen_texts_file', './test/test_gen.txt'])

        with open("./test/toy_data/eval_src.txt") as f:
            src_texts = [text.strip() for text in f]

        generate(args=args, src_texts=src_texts)

    def test_with_optional_args(self):
        parser = get_generation_parser()
        args = parser.parse_args(['-model_dir', './test/test_model',
                                  '-gen_texts_file', './test/test_gen.txt',
                                  '-infer_method', 'sample',
                                  '-sample_top_k', '0',
                                  '-sample_p', '0.7',
                                  '-sample_temperature', '1.0',
                                  '-require_src_in_gen',
                                  '-n_gen_per_src', '2',
                                  '-max_gen_attempts', '2',
                                  '-min_postproc_length', '2',
                                  '-max_postproc_length', '50',
                                  '-max_redundancy_rate', '0.9',
                                  '-block_repeat',
                                  '-require_paired_punct',
                                  '-require_eos_punct',
                                  '-block_quotes',
                                  # '-block_profanity'
                                  ])

        with open("./test/toy_data/eval_src.txt") as f:
            src_texts = [text.strip() for text in f]

        generate(args=args, src_texts=src_texts)

        print("Cleaning up...")
        shutil.rmtree("./test/test_model")
        os.remove("./test/test_gen.txt")


class Test01LMTrain(unittest.TestCase, BaseTestTrain):
    config_filename = "transformer_lm_config.json"


class Test02LMGenerate(unittest.TestCase, BaseTestGenerate):
    config_filename = "transformer_lm_config.json"


class Test03GPT2LMTrain(unittest.TestCase, BaseTestTrain):
    config_filename = "gpt2_lm_config.json"


class Test04GPT2LMGenerate(unittest.TestCase, BaseTestGenerate):
    config_filename = "gpt2_lm_config.json"


class Test05Seq2SeqTrain(unittest.TestCase, BaseTestTrain):
    config_filename = "transformer_seq2seq_config.json"


class Test06Seq2SeqGenerate(unittest.TestCase, BaseTestGenerate):
    config_filename = "transformer_seq2seq_config.json"


class Test07GPT2Seq2SeqTrain(unittest.TestCase, BaseTestTrain):
    config_filename = "gpt2_seq2seq_config.json"


class Test08GPT2Seq2SeqGenerate(unittest.TestCase, BaseTestGenerate):
    config_filename = "gpt2_seq2seq_config.json"


class Test09RNNSeq2SeqTrain(unittest.TestCase, BaseTestTrain):
    config_filename = "rnn_seq2seq_config.json"


class Test10RNNSeq2SeqGenerate(unittest.TestCase, BaseTestGenerate):
    config_filename = "rnn_seq2seq_config.json"


class Test11CopyRNNSeq2SeqTrain(unittest.TestCase, BaseTestTrain):
    config_filename = "copy_rnn_seq2seq_config.json"


class Test12CopyRNNSeq2SeqGenerate(unittest.TestCase, BaseTestGenerate):
    config_filename = "copy_rnn_seq2seq_config.json"


class Test13RNNLMTrain(unittest.TestCase, BaseTestTrain):
    config_filename = "rnn_lm_config.json"


class Test14RNNLMGenerate(unittest.TestCase, BaseTestGenerate):
    config_filename = "rnn_lm_config.json"


# class Test15CopyRNNLMTrain(unittest.TestCase, BaseTestTrain):
#     config_filename = "copy_rnn_lm_config.json"


# class Test16CopyRNNLMGenerate(unittest.TestCase, BaseTestGenerate):
#     config_filename = "copy_rnn_lm_config.json"


if __name__ == '__main__':
    unittest.main()
