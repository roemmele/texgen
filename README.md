# texgen

## Intro

Texgen is a library for neural text generation that is built on top of [Texar-PyTorch](https://github.com/asyml/texar-pytorch). Texar-PyTorch is already fairly high-level and easy to use for prototyping; texgen just provides the convenience of ready-to-go scripts for training and generation. It supports RNN and transformer-based language models (LM; decoder only) and sequence-to-sequence models (seq2seq; encoder-decoder). This library came about through experiments with sentence infilling/elaboration (see [here](https://github.com/roemmele/insentivize)), but it is generic enough that it could be adapted for any text generation task that relies on a LM or seq2seq model trained on source-target pairs. There is support for creating models initialized with pretrained weights (currently only loading from GPT-2 is available, but any pretrained model implemented by Texar-PyTorch could be added).

## Installation

This library can be installed by running "pip install ." here, at the top level of this repo (i.e. at the same level as the package folder texgen). Once you install, you should be able to "import texgen" outside this repo.

## Interaction

The library provides simple interfaces with the training and generation code. You can import texgen/train.py and texgen/generate.py inside scripts, or train_script.py and generation_script.py can be run directly from the command-line (see below). See the unittests file test/test.py for examples of how to run training and generation from inside a script. For command-line usage, see texgen/args.py for all command-line arguments that can be provided to the scripts. In line with Texar, texgen builds and loads models from config files encoded in JSON, which specify all the hyperparameters of a model. The format of these config files is consistent with the [scheme](https://texar-pytorch.readthedocs.io/en/latest/code/hyperparams.html) used by Texar, but texgen has just a few additional configuration fields that are specific to texgen and not recognized by Texar. Here is the format of a config file:

```

{
    "tokenizer": [TOKENIZER_NAME], # Required; choices=['gpt2-small', 'gpt2-medium', 'gpt2-large']
    "encoder": { # Only defined if using seq2seq (LM has no encoder)
        "type": [ARCHITECTURE_NAME], # Required; choices=['transformer', 'rnn']
        "tx_hparams": {} #If type="transformer", these are the Texar HParams allowed by GPT2Encoder; if type="rnn", these are the Texar HParams allowed by UnidirectionalRNNEncoder; see Texar documentation of these classes
    },
    "decoder": { # Required for both seq2seq and LM models
        "type": [ARCHITECTURE_NAME], # Required; choices=['transformer', 'rnn']
        "tx_hparams": {} #If type="transformer", these are the Texar HParams allowed by GPT2Decoder; if type="rnn", these are the Texar HParams allowed by RNNDecoderBase; see Texar documentation of these classes
    }
}
```

The Texar-supported hyperparameters are always defined under the field "tx_hparams" for each model component. Any hyperparameters that are not defined inside "tx_hparams" will be set to a default value defined by Texar. The Texar documentation defines the accepted values and defaults for each model class. See the notes above about how type dictates which Texar classes are used; the tx_hparams values being passed must be consistent with those defined for that class. Right now only a limited set of classes are supported ([GPT2Encoder<sup>\*</sup>](https://texar-pytorch.readthedocs.io/en/latest/code/modules.html#gpt2encoder) and [UnidirectionalRNNEncoder](https://texar-pytorch.readthedocs.io/en/latest/code/modules.html#unidirectionalrnnencoder) for the encoder; [GPT2Decoder<sup>\*</sup>](https://texar-pytorch.readthedocs.io/en/latest/code/modules.html#gpt2decoder) and [RNNDecoderBase](https://texar-pytorch.readthedocs.io/en/latest/code/modules.html#rnndecoderbase) for the decoder), but this can certainly be extended to any model class implemented by Texar.

For example here is the config file for a transformer LM (decoder only) model that uses weights from GPT2-small, which is the model used for sentence elaboration:

```
{
    "tokenizer": "gpt2-small",
    "decoder": {
        "type": "transformer",
        "tx_hparams": {
            "pretrained_model_name": "gpt2-small" # Set this to use pretrained model weights, or set to null to initialize weights from scratch; choices = ['null', 'gpt2-small', 'gpt2-medium', 'gpt2-large']
        }
    }
}
```

See test/test_configs for other examples of minimal config files for different models.

<sub><sup>\*</sup>The "regular" (i.e. non-pretrained) transformer encoder and decoder in texgen use the GPTEncoder and GPT2Decoder classes from Texar, which might be confusing because they are not using GPT2 weights. Texar's GPT2Encoder and GPT2Decoder classes refer to the architecture of GPT2, not the model weights. These classes are just implementations of the Transformer architecture combined with token and position embeddings, with the option to load pretrained weights when initializing. Texar also provides the TransformerEncoder and TransformerDecoder classes which are wrapped by GPT2Encoder and GPT2Decoder, respectively, but do not include token embedding layers. It is easier to just interface with the GPT2 classes for the transformer LM and seq2seq models supported by the library.</sub>

## Examples

### Training

You can run train_script.py on the command line in order to train a model. For example, to train the model defined in the config test/test_configs/gpt2_lm_config.json (transformer language model initialized with GPT-2 weights) on a tiny dataset of source and target texts in test/test_data/ and save the resulting model files to the directory test/test_model, here is the minimal command:

```
python train_script.py -train_src_file test/toy_data/train_src.txt -train_tgt_file test/toy_data/train_tgt.txt -eval_src_file test/toy_data/eval_src.txt -eval_tgt_file test/toy_data/eval_tgt.txt -config_file test/test_configs/gpt2_lm_config.json -save_dir test/test_model -valid_epoch_end
```

Run `python train_script.py -h` to see a description of all command-line parameters that can be specified when running this script (also defined in texgen/args.py).

The saved model files consist of a .pt checkpoint named according to its timestamp, a file with metadata about the checkpoint, and a hparams.json file that contains the same info as the config file used to create the model.

### Generation

To apply a trained model to generating target outputs for source inputs, you can run generation_script.py. For example, using the toy dataset and model trained above, run:

```
python generation_script.py -model_dir test/test_model/ -src_texts_file test/toy_data/eval_src.txt -gen_texts_file test/test_gen.txt
```

In this case, the generated texts for each source input will appear in test/test_gen.txt (note that if you run this with the toy model trained above, you will likely not see meaningful output). Run `python generation_script.py -h` to see the description of all command-line parameters that can be specified, as defined in texgen/args.py.
