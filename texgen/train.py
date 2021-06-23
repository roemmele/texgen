import warnings
warnings.filterwarnings('ignore')

import os
import logging
import pickle
import importlib
import functools
from pathlib import Path
from argparse import Namespace

import torch
from texar.torch.run import Executor, metric, cond, action
from texar.torch.run.metric.summary import Average, RunningAverage
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import LambdaLR
from typing import Any, Callable, List, Optional, Tuple

from .data import create_texar_dataset, load_tokenizer, TexarDataset
from .construct import create_model, load_hparams
from .metric import RawMetric
from .args import get_train_parser
from .model.lm import LM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)

logger = logging.getLogger(__name__)


def get_lr_scaler(step: int,
                  warmup_steps: int):
    """Noam learning rate schedule described in
    https://arxiv.org/pdf/1706.03762.pdf.
    """
    if step <= warmup_steps:
        lr_scaler = min(1.0, step / warmup_steps)  # Linear warmup
    else:
        lr_scaler = 0.9 ** (step / warmup_steps)  # Exponential decay
    return lr_scaler


def create_optimizer_and_scheduler(model: LM,
                                   learning_rate: float=0.001,
                                   dynamic_lr: bool=False,
                                   warmup_steps: int=4000
                                   ) -> Tuple[Adam, LambdaLR]:
    # Define optimizer and scheduler
    if dynamic_lr:
        lr_scaler_fn = functools.partial(get_lr_scaler, warmup_steps=warmup_steps)
    else:
        lr_scaler_fn = lambda _: 1.0

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 betas=(0.9, 0.998), eps=1e-9)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scaler_fn)
    return optimizer, lr_scheduler


def load_metrics(pg_metric_names: List[Any],
                 eval_metric_names: List[Any],
                 text_decode_fn: Callable
                 ) -> Tuple[List[Any], List[Any]]:
    metric_fns_map = {metric_name:
                      RawMetric.get_metric_from_name(metric_name)(decode_fn=text_decode_fn)
                      for metric_name in set(pg_metric_names + eval_metric_names)}

    pg_metric_fns = [metric_fns_map[metric_name] for metric_name in pg_metric_names]
    eval_metric_fns = [metric_fns_map[metric_name] for metric_name in eval_metric_names]

    return pg_metric_fns, eval_metric_fns


def create_texar_metrics(pg_metric_fns: List[Any],
                         eval_metric_fns: List[Any],
                         log_iterations: int=100
                         ) -> Tuple[List[RunningAverage], List[Average]]:

    train_metrics = [metric.RunningAverage(log_iterations),
                     metric.RunningAverage(log_iterations,
                                           pred_name='mle_loss'),
                     *(metric.RunningAverage(log_iterations,
                                             pred_name='{}_loss'.format(metric_fn.metric_name))
                       for metric_fn in pg_metric_fns)]

    if eval_metric_fns:
        eval_metrics = [*(metric.Average(pred_name=metric_fn.metric_name,
                                         higher_is_better=metric_fn.higher_is_better)
                          for metric_fn in eval_metric_fns),
                        metric.Average(pred_name='mle_loss')]
    else:
        eval_metrics = [metric.Average(pred_name='mle_loss')]

    return train_metrics, eval_metrics


def create_executor(model: LM,
                    save_dir: str,
                    train_data: TexarDataset,
                    eval_data: TexarDataset,
                    optimizer: Adam,
                    lr_scheduler: LambdaLR,
                    load_from_dir: Optional[str]=None,
                    train_metrics: List[RunningAverage]=[],
                    eval_metrics: List[Average]=[],
                    max_grad_norm: float=5.0,
                    accum_steps: int=1,
                    max_epochs: int=100,
                    patience: int=2,
                    valid_iterations: int=1000,
                    valid_epoch_end: bool=False,
                    log_iterations: int=100
                    ) -> Executor:

    # Use Executor API to train and validate
    executor = Executor(model=model,
                        train_data=train_data,
                        valid_data=eval_data,
                        test_data=eval_data,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        grad_clip=max_grad_norm,
                        num_iters_per_update=accum_steps,
                        stop_training_on=cond.epoch(max_epochs),
                        log_every=cond.iteration(log_iterations),
                        validate_mode='eval',
                        test_mode='eval',
                        validate_every=([cond.epoch(1),
                                         cond.iteration(valid_iterations)] if valid_epoch_end
                                        else [cond.iteration(valid_iterations)]),
                        train_metrics=train_metrics + [metric.LR(optimizer)],
                        valid_metrics=eval_metrics,
                        save_every=cond.validation(better=True),
                        plateau_condition=[
                            cond.consecutive(cond.validation(better=False), 1)],
                        action_on_plateau=[
                            action.early_stop(patience=patience)],
                        max_to_keep=1,
                        checkpoint_dir=save_dir)

    if load_from_dir is not None:
         # Seems like a bug where checkpoint is only looked for in executor.checkpoint_dir
         # Get around this by temporarily changing checkpoint_dir to args.load_from_dir then changing it back to args.save_dir
        executor.checkpoint_dir = Path(load_from_dir)
        loaded_checkpoint = executor.load()
        logger.info("Initialized model from checkpoint {}".format(loaded_checkpoint))
        executor.checkpoint_dir = Path(save_dir)

    return executor


def train(args: Namespace,
          train_src_texts: List[str],
          train_tgt_texts: List[str],
          eval_src_texts: List[str],
          eval_tgt_texts: List[str],
          train_ref_texts: Optional[List[str]]=None,
          eval_ref_texts: Optional[List[str]]=None
          ) -> None:

    if args.load_from_dir:
        logger.info(("Loading model configuration from {}. " +
                     "All hyperparameter settings will be read from here " +
                     "and will override any settings provided as command-line arguments.").format(
            os.path.join(args.load_from_dir)))
        model_hparams = load_hparams(path=args.load_from_dir)

    else:
        assert args.config_file is not None
        model_hparams = load_hparams(path=args.config_file)

    tokenizer = load_tokenizer(model_hparams['tokenizer'])

    # Make dataset objects for Texar API
    texar_train_data, texar_eval_data = create_texar_dataset(
        tokenizer=tokenizer,
        train_src_texts=train_src_texts,
        train_tgt_texts=train_tgt_texts,
        eval_src_texts=eval_src_texts,
        eval_tgt_texts=eval_tgt_texts,
        train_ref_texts=train_ref_texts,
        eval_ref_texts=eval_ref_texts,
        max_src_length=args.max_src_length,
        max_tgt_length=args.max_tgt_length,
        batch_size=args.batch_size
    )

    # Initialize the policy gradient and evaluation metrics based on given metric names
    pg_metric_fns, eval_metric_fns = load_metrics(
        pg_metric_names=args.pg_metrics,
        eval_metric_names=args.eval_metrics,
        text_decode_fn=(lambda text:
                        tokenizer.map_id_to_text(text, skip_special_tokens=True))
    )

    # Initialize model with properties and policy gradient/evaluation functions
    model = create_model(hparams=model_hparams,
                         save_dir=args.save_dir)
    model.set_pg_and_eval_fns(pg_metric_fns, eval_metric_fns)

    # Create optimizer and LR scheduler
    optimizer, lr_scheduler = create_optimizer_and_scheduler(model,
                                                             learning_rate=args.learning_rate,
                                                             dynamic_lr=args.dynamic_lr,
                                                             warmup_steps=args.warmup_steps)

    # Wrap metrics inside Texar Metric API to be used by Executor
    (texar_train_metrics,
     texar_eval_metrics) = create_texar_metrics(pg_metric_fns=pg_metric_fns,
                                                eval_metric_fns=eval_metric_fns,
                                                log_iterations=args.log_iterations)

    # Create Texar Executor
    executor = create_executor(model=model,
                               save_dir=args.save_dir,
                               train_data=texar_train_data,
                               eval_data=texar_eval_data,
                               optimizer=optimizer,
                               lr_scheduler=lr_scheduler,
                               load_from_dir=args.load_from_dir,
                               train_metrics=texar_train_metrics,
                               eval_metrics=texar_eval_metrics,
                               max_grad_norm=args.max_grad_norm,
                               accum_steps=args.accum_steps,
                               max_epochs=args.max_epochs,
                               patience=args.patience,
                               valid_iterations=args.valid_iterations,
                               valid_epoch_end=args.valid_epoch_end,
                               log_iterations=args.log_iterations)

    logger.info("Validation result prior to training:")
    executor.test()

    executor.train()
