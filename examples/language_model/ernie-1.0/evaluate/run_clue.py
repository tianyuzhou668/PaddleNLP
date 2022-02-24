# Copyright (c) 2021s PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import sys
import random
import time
import math
from functools import partial
import distutils.util

import numpy as np
import paddle
from paddle.io import DataLoader
import paddle.nn as nn

from paddle.metric import Accuracy

import paddlenlp
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad, Dict
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
from paddlenlp.transformers import LinearDecayWithWarmup

from paddlenlp.utils.log import logger

from data import convert_clue

METRIC_CLASSES = {
    "afqmc": Accuracy,
    "tnews": Accuracy,
    "iflytek": Accuracy,
    "ocnli": Accuracy,
    "cmnli": Accuracy,
    "cluewsc2020": Accuracy,
    "csl": Accuracy,
}

TokenClassificationModels = []
SequenceClassificationModels = []

for x in dir(paddlenlp.transformers):
    if x.endswith("ForTokenClassification"):
        if not x.startswith("AutoModel"):
            TokenClassificationModels.append(x)

    elif x.endswith("ForSequenceClassification"):
        if not x.startswith("AutoModel"):
            SequenceClassificationModels.append(x)

# print(SequenceClassificationModels)
#
# for x in SequenceClassificationModels:
#     for c in [getattr(paddlenlp.transformers, x)]:
#         print(list(c.pretrained_init_configuration.keys()))


def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--dataset",
        default=None,
        type=str,
        required=True,
        help="The name of the dataset to train selected in the list: " +
        ", ".join(METRIC_CLASSES.keys()), )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=False,
        help="The name of the task to train, such as clue task: " +
        ", ".join(METRIC_CLASSES.keys()), )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(
            sum([
                list(classes.pretrained_init_configuration.keys())
                for classes in [
                    getattr(paddlenlp.transformers, x)
                    for x in SequenceClassificationModels
                ]
            ], [])), )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.", )

    parser.add_argument(
        "--learning_rate",
        default=1e-4,
        type=float,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size per GPU/CPU for training.", )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some.")

    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        help="Total number of training epochs to perform.", )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Log every X updates steps.")
    parser.add_argument(
        "--valid_steps",
        type=int,
        default=100,
        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )

    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Linear warmup over warmup_steps. If > 0: Override warmup_proportion"
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Linear warmup proportion over total steps.")

    parser.add_argument(
        "--use_amp",
        type=distutils.util.strtobool,
        default=False,
        help="Enable mixed precision training.")
    parser.add_argument(
        "--scale_loss",
        type=float,
        default=2**15,
        help="The value of scale_loss for fp16.")

    parser.add_argument(
        "--adam_epsilon",
        default=1e-6,
        type=float,
        help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="The max value of grad norm.")

    parser.add_argument(
        "--seed", default=42, type=int, help="random seed for initialization")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="The device to select to train the model, is must be cpu/gpu/xpu.")
    args = parser.parse_args()
    return args


def set_seed(args):
    # Use the same data seed(for data shuffle) for all procs to guarantee data
    # consistency after sharding.
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Maybe different op seeds(for dropout) for different procs is better. By:
    # `paddle.seed(args.seed + paddle.distributed.get_rank())`
    paddle.seed(args.seed)


@paddle.no_grad()
def evaluate(model, loss_fct, metric, data_loader):
    model.eval()
    metric.reset()
    for batch in data_loader:
        input_ids, segment_ids, labels = batch
        logits = model(input_ids, segment_ids)
        loss = loss_fct(logits, labels)
        correct = metric.compute(logits, labels)
        metric.update(correct)
    res = metric.accumulate()
    logger.info("eval loss: %f, acc: %s, " % (loss.numpy(), res))
    model.train()
    metric.reset()
    return res


def create_dataloader(args, trans_func, batchify_fn, modes):
    train_ds = load_dataset(
        args.dataset, args.task_name, splits=mode, lazy=False)
    train_ds = train_ds.map(trans_func, lazy=True)
    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_ds, batch_size=args.batch_size, shuffle=True)

    train_data_loader = DataLoader(
        dataset=train_ds,
        batch_sampler=train_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=True)
    return train_data_loader


def do_train(args):
    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args)

    args.task_name = args.task_name.lower()
    metric_class = METRIC_CLASSES[args.task_name]

    train_ds = load_dataset(
        args.dataset, args.task_name, splits="train", lazy=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    trans_func = partial(
        convert_clue,
        tokenizer=tokenizer,
        label_list=train_ds.label_list,
        max_seq_length=args.max_seq_length)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
        Stack(dtype="int64" if train_ds.label_list else "float32")  # label
    ): fn(samples)

    train_data_loader = create_dataloader(args, trans_func, batchify_fn,
                                          "train")
    dev_data_loader = create_dataloader(args, trans_func, batchify_fn, "dev")

    num_classes = 1 if train_ds.label_list == None else len(train_ds.label_list)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, num_classes=num_classes)

    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    if args.max_steps > 0:
        num_training_steps = args.max_steps
        num_train_epochs = math.ceil(num_training_steps /
                                     len(train_data_loader))
    else:
        num_training_steps = len(train_data_loader) * args.num_train_epochs
        num_train_epochs = args.num_train_epochs

    warmup = args.warmup_steps if args.warmup_steps > 0 else args.warmup_proportion

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         warmup)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        beta1=0.9,
        beta2=0.999,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params,
        grad_clip=nn.ClipGradByGlobalNorm(args.max_grad_norm))

    loss_fct = paddle.nn.loss.CrossEntropyLoss(
    ) if train_ds.label_list else paddle.nn.loss.MSELoss()

    metric = metric_class()
    if args.use_amp:
        scaler = paddle.amp.GradScaler(init_loss_scaling=args.scale_loss)

    best_acc = 0.0
    global_step = 0
    tic_train = time.time()

    for epoch in range(num_train_epochs):
        for step, batch in enumerate(train_data_loader):
            global_step += 1
            input_ids, segment_ids, labels = batch
            with paddle.amp.auto_cast(
                    args.use_amp,
                    custom_white_list=["layer_norm", "softmax", "gelu"], ):
                logits = model(input_ids, segment_ids)
                loss = loss_fct(logits, labels)

            probs = F.softmax(logits, axis=1)
            correct = metric.compute(probs, labels)
            metric.update(correct)
            acc = metric.accumulate()

            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.minimize(optimizer, loss)
            else:
                loss.backward()
                optimizer.step()

            lr_scheduler.step()
            optimizer.clear_grad()

            if global_step % args.logging_steps == 0:
                logger.info(
                    "global step %d/%d, epoch: %d, batch: %d, rank_id: %s, loss: %f, lr: %.10f, speed: %.4f step/s"
                    % (global_step, num_training_steps, epoch, step, rank, loss,
                       optimizer.get_lr(),
                       args.logging_steps / (time.time() - tic_train)))
                tic_train = time.time()
            if global_step % args.valid_steps == 0 or global_step == num_training_steps:
                tic_eval = time.time()
                acc = evaluate(model, loss_fct, metric, dev_data_loader)
                logger.info("eval done total : %s s" % (time.time() - tic_eval))
                if acc > best_acc:
                    best_acc = acc
            if global_step >= num_training_steps:
                logger.info("best_acc: {:.6f}".format(best_acc))
                return

    logger.info("best_acc: {:.6f}".format(best_acc))


def print_arguments(args):
    """print arguments"""
    logger.info('{:^40}'.format("Configuration Arguments"))
    logger.info('{:20}:{}'.format("paddle commit id", paddle.version.commit))
    for arg in vars(args):
        logger.info('{:20}:{}'.format(arg, getattr(args, arg)))


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)
    do_train(args)
