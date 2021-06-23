import os
import json
import logging
import torch
from .model import LM, Seq2SeqModel
from texgen.model.lm import LM
from typing import Dict, Tuple, Union, Any

logger = logging.getLogger(__name__)

default_hparams_filename = "hparams.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_hparams(path: str) -> Dict[str, Any]:
    # If only directory given, look for config file with default filename.
    # Otherwise assume path is path to config file
    filepath = path
    if os.path.isdir(path):
        filepath = os.path.join(path, default_hparams_filename)
    with open(filepath) as f:
        hparams = json.load(f)
    return hparams


def save_hparams(hparams: Dict[str, Any],
                 save_dir: str
                 ) -> None:
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    with open(os.path.join(save_dir, default_hparams_filename), 'w') as f:
        json.dump(hparams, f)
    logger.info("Saved model configuration to {}".format(
        os.path.join(save_dir, default_hparams_filename)))


def init_model_class(hparams: Dict[str, Any]) -> LM:
    if 'encoder' in hparams:  # If encoder params specified, model design is Seq2Seq
        model = Seq2SeqModel(encoder_type=hparams['encoder']['type'],
                             encoder_hparams=hparams['encoder']['tx_hparams'],
                             decoder_type=hparams['decoder']['type'],
                             decoder_hparams=hparams['decoder']['tx_hparams'],
                             use_copy_mechanism=hparams['decoder'].get('use_copy_mechanism', False),
                             vocab_size=hparams.get('token_embedder', {}).get('vocab_size'),
                             embedder_hparams=hparams.get('token_embedder', {}).get('tx_hparams'))
    else:  # No encoder params, model design is LM
        model = LM(architecture_type=hparams['decoder']['type'],
                   vocab_size=hparams.get('token_embedder', {}).get('vocab_size'),
                   emb_hparams=hparams.get('token_embedder', {}).get('tx_hparams'),
                   model_hparams=hparams['decoder']['tx_hparams'])
    return model


def create_model(hparams: Dict[str, Any],
                 save_dir: str
                 ) -> LM:
    model = init_model_class(hparams)
    model.to(device)
    save_hparams(hparams, save_dir)
    logger.info("Created {} model".format(model.__class__.__name__))
    logger.info("# trainable parameters in model: {}".format(
        sum(param.numel() for param in model.parameters()
            if param.requires_grad)))
    return model


def load_model(load_dir: str) -> Tuple[LM, Dict[str, Any]]:
    hparams = load_hparams(load_dir)
    model = init_model_class(hparams)
    checkpoint_filepath = sorted([os.path.join(load_dir, filename)
                                  for filename in os.listdir(load_dir)
                                  if filename.endswith('.pt')])[-1]  # Load most recently saved checkpoint
    checkpoint = torch.load(checkpoint_filepath, map_location=device)
    model.load_state_dict(checkpoint.model)
    model.to(device)
    logger.info("Loaded {} model from {}".format(model.__class__.__name__, checkpoint_filepath))
    return model, hparams
