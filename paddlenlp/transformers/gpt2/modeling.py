# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import collections
from collections import namedtuple
import math

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.tensor as tensor
from paddle.fluid import layers
from paddle.fluid.framework import in_dygraph_mode
from paddle.nn.layer.transformer import _convert_param_attr_to_list
from paddle.fluid.initializer import Normal, Constant, NumpyArrayInitializer
import paddle.distributed.fleet as fleet

from .. import PretrainedModel, register_base_model

__all__ = [
    'GPT2Model',
    "GPT2PretrainedModel",
    'GPT2ForPretraining',
    'GPT2PretrainingCriterion',
    'GPT2ForGreedyGeneration',
    'GPT2ForTopKPGeneration',
]


def print_t(name, tensor):
    print(name)
    print("sum", tensor.sum())
    print("abs sum", tensor.abs().sum())
    print(tensor)


GroupInfo = namedtuple('GroupInfo', ['size', 'rank', 'world'])


class Topology:
    def __init__(self, rank, world_size, dp, pp, sharding, mp):
        arr = np.arange(0, dp * pp * sharding * mp).reshape(
            [dp, pp, sharding, mp])
        idp, ipp, isharding, imp = np.where(arr == rank)
        idp = idp[0]
        ipp = ipp[0]
        isharding = isharding[0]
        imp = imp[0]
        self.world = GroupInfo(
            size=world_size, rank=rank, world=list(range(0, world_size)))
        mp = arr[idp, ipp, isharding, :]
        self.mp = GroupInfo(size=len(mp), rank=imp, world=mp.tolist())
        sharding = arr[idp, ipp, :, imp]
        self.sharding = GroupInfo(
            size=len(sharding), rank=isharding, world=sharding.tolist())
        pp = arr[idp, :, isharding, imp]
        self.pp = GroupInfo(size=len(pp), rank=ipp, world=pp.tolist())
        dp = arr[:, ipp, isharding, imp]
        self.dp = GroupInfo(size=len(dp), rank=idp, world=dp.tolist())

    def __repr__(self):
        return f'dp: {self.dp}, pp: {self.pp}, sharding: {self.sharding}, mp: {self.mp}'


class ParallelEmbedding(nn.Layer):
    """
    Parallel Embedding
    """

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 padding_idx=None,
                 weight_attr=None,
                 name=None):
        super().__init__()
        size = (num_embeddings, embedding_dim)
        if in_dygraph_mode():
            rank = paddle.distributed.get_rank()
            nranks = paddle.distributed.get_world_size()
        else:
            # assert fleet._role_maker, ("To use paddle.distributed.split, "
            #                            "you must call fleet.init() firstly.")
            rank = fleet.worker_index()
            nranks = fleet.worker_num()

        # rank within a model parallel group
        inner_rank = rank % num_partitions
        self.inner_rank = inner_rank
        self.num_partitions = num_partitions

        assert axis == 0, ("We only support to split the weight of embedding "
                           "along the first axis now.")
        per_part_size = (size[0] + num_partitions - 1) // num_partitions
        last_part_size = size[0] - per_part_size * (num_partitions - 1)
        if inner_rank == num_partitions - 1: per_part_size = last_part_size
        per_part_size += 1  # make the last row as the padding index

        origin_size = size
        if not name:
            name = "emb_rank_%d" % inner_rank
        else:
            name = name + "_rank_%d" % inner_rank

        self.per_part_embeddings = per_part_size
        self.origin_num_embeddings = origin_size[0]
        self.embedding = paddle.nn.Embedding(
            per_part_embeddings,
            origin_size[1],
            padding_idx=per_part_embeddings - 1,
            sparse=False,
            weight_attr=param_attr,
            name=name)

        self.embedding.weight.is_distributed = True

        startup_block = paddle.static.default_startup_program().global_block()
        main_block = paddle.static.default_main_program().global_block()
        startup_block.vars[embedding.weight.name].is_distributed = True
        main_block.vars[embedding.weight.name].is_distributed = True

    def forward(x):
        origin_input_shape = x.shape
        if len(origin_input_shape) == 2:
            x = paddle.unsqueeze(x, axis=-1)
        else:
            assert origin_input_shape[-1] == 1, (
                "The last dimension size of x must be 1.")
        x_shard = paddle.shard_index(x, self.origin_num_embeddings,
                                     self.num_partitions, self.inner_rank,
                                     self.per_part_embeddings - 1)
        if len(origin_input_shape) == 2:
            x_shard = paddle.squeeze(x_shard, axis=-1)

        emb_out = self.embedding(x_shard)
        paddle.distributed.all_reduce(emb_out, group=None)
        return emb_out


class ParallelLinear(nn.Layer):
    """
    Parallel Linear
    """

    def __init__(self,
                 size,
                 axis,
                 num_partitions=1,
                 gather_out=True,
                 param_attr=None,
                 bias_attr=None,
                 name=None):
        super().__init__()
        if in_dygraph_mode():
            rank = paddle.distributed.get_rank()
            nranks = paddle.distributed.get_world_size()
        else:
            #assert fleet._role_maker, ("To use paddle.distributed.split, "
            #                           "you must call fleet.init() firstly.")
            rank = fleet.worker_index()
            nranks = fleet.worker_num()

        # rank within a model parallel group
        inner_rank = rank % num_partitions
        self.axis = axis
        if axis == 0:
            assert size[0] % num_partitions == 0, (
                "Number of rows of the weight for linear ({}) must be"
                " divisible by num_partitions ({})".format(size[0],
                                                           num_partitions))
            self.per_part_size = size[0] // num_partitions
            linear_size = (self.per_part_size, size[1])
            # assert x.shape[-1] == self.per_part_size, (
            #     "The width ({}) of the input "
            #     "x must be equal to the height ({}) of the weight. Maybe you "
            #     "should split the input x using paddle.split.".format(
            #         x.shape[-1], self.per_part_size))

        elif axis == 1:
            assert size[1] % num_partitions == 0, (
                "Number of column of the weight for linear ({}) must be"
                " divisible by num_partitions ({})".format(size[1],
                                                           num_partitions))
            self.per_part_size = size[1] // num_partitions
            linear_size = (size[0], self.per_part_size)
        else:
            raise ValueError("The value of axis must be 0 or 1, but the value "
                             "given is {}.".format(axis))

        num_rows, num_cols = linear_size

        self.gather_out = gather_out
        self.axis = axis
        if not name:
            name = "fc_by_row_rank_%d" % inner_rank if axis == 0 else "fc_by_col_rank_%d" % inner_rank
        else:
            name = name + "_by_row_rank_%d" % inner_rank if axis == 0 else name + "_by_col_rank_%d" % inner_rank
        self.linear = paddle.nn.Linear(
            num_rows,
            num_cols,
            weight_attr=param_attr,
            bias_attr=bias_attr,
            name=name)

        weight = self.linear.weight
        weight.is_distributed = True
        startup_block = paddle.static.default_startup_program().global_block()
        main_block = paddle.static.default_main_program().global_block()
        startup_block.vars[weight.name].is_distributed = True
        main_block.vars[weight.name].is_distributed = True

    def forward(self, x):
        if self.axis == 0:
            assert x.shape[-1] == self.per_part_size, (
                "The width ({}) of the input "
                "x must be equal to the height ({}) of the weight. Maybe you "
                "should split the input x using paddle.split.".format(
                    x.shape[-1], self.per_part_size))

        linear_out = self.linear(x)
        if self.gather_out:
            if self.axis == 0:
                paddle.distributed.all_reduce(linear_out)
            else:
                output = []
                paddle.distributed.all_gather(output, linear_out)
                linear_out = paddle.concat(
                    output, axis=len(linear_out.shape) - 1)
        return linear_out


class ColumnParallelLiner(ParallelLinear):
    def __init__(self,
                 size,
                 num_partitions,
                 param_attr=None,
                 bias_attr=None,
                 name=None):
        super().__init__(
            size,
            axis=1,
            num_partitions=num_partitions,
            gather_out=False,
            param_attr=param_attr,
            bias_attr=bias_attr)


class RowParallelLiner(ParallelLinear):
    def __init__(self,
                 size,
                 num_partitions,
                 param_attr=None,
                 bias_attr=None,
                 name=None):
        super().__init__(
            size,
            axis=0,
            num_partitions=num_partitions,
            gather_out=True,
            param_attr=param_attr,
            bias_attr=bias_attr)


# def _beuild_linear_column_parallel(x, n_in, n_out, name, init, mp):
#     return paddle.distributed.split(x,
#             size=(n_in, n_out),
#             operation='linear',
#             axis=1,
#             gather_out=False,
#             num_partitions=mp,
#             weight_attr=fluid.ParamAttr(
#                     name='%s.w_0' % name if name is not None else None,
#                     initializer=init),
#             bias_attr='%s.b_0' % name if name is not None else None, )
# 
# 
# def _build_linear_row_parallel(x, n_in, n_out, name, init, mp):
#     return paddle.distributed.split(x,
#             size=(n_in, n_out),
#             operation='linear',
#             axis=0,
#             gather_out=True,
#             num_partitions=mp,
#             weight_attr=fluid.ParamAttr(
#                     name='%s.w_0' % name if name is not None else None,
#                     initializer=init),
#             bias_attr='%s.b_0' % name if name is not None else None, )
# 


class LayerNormByBN(nn.Layer):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        # normalized_shape = np.prod(input_shape[:-1])
        normalized_shape = 16 * 1024
        input_shape = [dim if dim != -1 else 1024]
        # print(input_shape)
        self.batch_norm = paddle.nn.BatchNorm(
            normalized_shape,
            epsilon=epsilon,
            param_attr=paddle.fluid.ParamAttr(),
            data_layout='NHWC',
            momentum=0.0,
            do_model_average_for_mean_and_var=False)

        # self.batch_norm.weight.stop_gradient =True
        # self.batch_norm.bias.stop_gradient =True

        self.weight = self.create_parameter(
            shape=[input_shape[-1]],
            dtype=self._dtype,
            default_initializer=Constant(1.0))
        self.bias = self.create_parameter(
            shape=[input_shape[-1]],
            dtype=self._dtype,
            default_initializer=Constant(0.0))

    def forward(self, input):
        in_shape = paddle.shape(input)
        # paddle.fluid.layers.Print(self.batch_norm.bias)
        # input = input.reshape([1, -1, in_shape[-1]])
        input = input.reshape([1, 16 * 1024, in_shape[-1]])
        input_trans = input.transpose([0, 2, 1])
        out = self.batch_norm(input_trans)
        out = out.squeeze(axis=0)
        out = out.transpose([1, 0])
        out = out * self.weight + self.bias
        out = out.reshape(in_shape)
        return out


# nn.LayerNorm = LayerNormByBN


class MultiHeadAttention(nn.Layer):
    """
    Attention mapps queries and a set of key-value pairs to outputs, and
    Multi-Head Attention performs multiple parallel attention to jointly attending
    to information from different representation subspaces.

    """

    Cache = collections.namedtuple("Cache", ["k", "v"])
    StaticCache = collections.namedtuple("StaticCache", ["k", "v"])

    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.,
                 kdim=None,
                 vdim=None,
                 need_weights=False,
                 weight_attr=None,
                 bias_attr=None,
                 topo=None):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.need_weights = need_weights

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if topo is None or toppo.mp.size == 1:
            self.q_proj = nn.Linear(
                embed_dim, embed_dim, weight_attr, bias_attr=bias_attr)
            self.k_proj = nn.Linear(
                self.kdim, embed_dim, weight_attr, bias_attr=bias_attr)
            self.v_proj = nn.Linear(
                self.vdim, embed_dim, weight_attr, bias_attr=bias_attr)
            self.out_proj = nn.Linear(
                embed_dim, embed_dim, weight_attr, bias_attr=bias_attr)
        else:
            self.q_proj = ColumnParallelLiner(
                (embed_dim, embed_dim),
                topo.mp.size,
                weight_attr,
                bias_attr=bias_attr)
            self.k_proj = ColumnParallelLiner(
                (self.kdim, embed_dim),
                topo.mp.size,
                weight_attr,
                bias_attr=bias_attr)
            self.v_proj = ColumnParallelLiner(
                (self.vdim, embed_dim),
                topo.mp.size,
                weight_attr,
                bias_attr=bias_attr)
            self.out_proj = RowParallelLiner(
                (embed_dim, embed_dim),
                topo.mp.size,
                weight_attr,
                bias_attr=bias_attr)

    def _prepare_qkv(self, query, key, value, use_cache=False, cache=None):
        r"""
        Prapares linear projected queries, keys and values for usage of subsequnt
        multiple parallel attention. If `cache` is not None, using cached results
        to reduce redundant calculations.

        """

        q = self.q_proj(query)
        q = tensor.reshape(x=q, shape=[0, 0, self.num_heads, self.head_dim])
        q = tensor.transpose(x=q, perm=[0, 2, 1, 3])

        if isinstance(cache, self.StaticCache):
            # for encoder-decoder attention in inference and has cached
            k, v = cache.k, cache.v
        else:
            k, v = self.compute_kv(key, value)

        if isinstance(cache, self.Cache):
            # for decoder self-attention in inference
            k = tensor.concat([cache.k, k], axis=2)
            v = tensor.concat([cache.v, v], axis=2)
        if use_cache is True:
            cache = self.Cache(k, v)

        return (q, k, v) if use_cache is False else (q, k, v, cache)

    def compute_kv(self, key, value):
        r"""
        Applies linear projection on input keys and values, then splits heads
        (reshape and transpose) to get keys and values from different representation
        subspaces. The results are used as key-values pairs for subsequent multiple
        parallel attention.

        It is part of calculations in multi-head attention, and is provided as
        a method to pre-compute and prefetch these results, thus we can use them
        to construct cache for inference.

        """
        k = self.k_proj(key)
        v = self.v_proj(value)
        k = tensor.reshape(x=k, shape=[0, 0, self.num_heads, self.head_dim])
        k = tensor.transpose(x=k, perm=[0, 2, 1, 3])
        v = tensor.reshape(x=v, shape=[0, 0, self.num_heads, self.head_dim])
        v = tensor.transpose(x=v, perm=[0, 2, 1, 3])
        return k, v

    def gen_cache(self, key, value=None, type=Cache):
        """
        Generates cache for `forward` usage in inference accroding to arguments.
        The generated cache is an instance of `MultiHeadAttention.Cache` or an
        instance of `MultiHeadAttention.StaticCache`.
        """
        if type == MultiHeadAttention.StaticCache:  # static_kv
            k, v = self.compute_kv(key, value)
            return self.StaticCache(k, v)
        elif value is None:  # incremental_state
            k = layers.fill_constant_batch_size_like(
                input=key,
                shape=[-1, self.num_heads, 0, self.head_dim],
                dtype=key.dtype,
                value=0)
            v = layers.fill_constant_batch_size_like(
                input=key,
                shape=[-1, self.num_heads, 0, self.head_dim],
                dtype=key.dtype,
                value=0)
            return self.Cache(k, v)
        else:
            # incremental_state with initial value, mainly for usage like UniLM
            return self.Cache(key, value)

    def forward(self,
                query,
                key,
                value,
                attn_mask=None,
                use_cache=False,
                cache=None):
        r"""
        Applies multi-head attention to map queries and a set of key-value pairs
        to outputs.
        """
        key = query if key is None else key
        value = query if value is None else value
        # compute q ,k ,v
        if use_cache is False:
            q, k, v = self._prepare_qkv(query, key, value, use_cache, cache)
        else:
            q, k, v, cache = self._prepare_qkv(query, key, value, use_cache,
                                               cache)
        # scale dot product attention
        product = layers.matmul(
            x=q, y=k, transpose_y=True, alpha=self.head_dim**-0.5)
        if attn_mask is not None:
            # product = product + attn_mask
            product = product * attn_mask
            mask_score = (attn_mask - 1.0) * 10000.0
            product = product + mask_score
        weights = F.softmax(product)
        if self.dropout:
            weights = F.dropout(
                weights,
                self.dropout,
                training=self.training,
                mode="upscale_in_train")

        out = tensor.matmul(weights, v)

        # combine heads
        out = tensor.transpose(out, perm=[0, 2, 1, 3])
        out = tensor.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # project to output
        out = self.out_proj(out)

        outs = [out]
        if self.need_weights:
            outs.append(weights)
        if use_cache:
            outs.append(cache)
        return out if len(outs) == 1 else tuple(outs)


class TransformerDecoder(nn.Layer):
    """
    TransformerDecoder is a stack of N decoder layers.
    """

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.LayerList([(
            decoder_layer
            if i == 0 else type(decoder_layer)(**decoder_layer._config))
                                    for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.checkpoints = []

    def forward(self,
                tgt,
                memory,
                tgt_mask=None,
                memory_mask=None,
                use_cache=False,
                cache=None):
        r"""
        Applies a stack of N Transformer decoder layers on inputs. If `norm` is
        provided, also applies layer normalization on the output of last decoder
        layer.
        """
        output = tgt
        new_caches = []
        self.checkpoints = []
        for i, mod in enumerate(self.layers):
            #print_t("loop_{}, output".format(i), output)
            # need devices guard.
            if cache is None:
                if use_cache:
                    output, new_cache = mod(output,
                                            memory,
                                            tgt_mask=tgt_mask,
                                            use_cache=use_cache,
                                            cache=cache)
                    new_caches.append(new_cache)
                else:
                    output = mod(output,
                                 memory,
                                 tgt_mask=tgt_mask,
                                 use_cache=use_cache,
                                 cache=cache)

            else:
                output, new_cache = mod(output,
                                        memory,
                                        tgt_mask=tgt_mask,
                                        use_cache=use_cache,
                                        cache=cache[i])
                new_caches.append(new_cache)
            self.checkpoints.append(output.name)

        if self.norm is not None:
            output = self.norm(output)
        return output if use_cache is False else (output, new_caches)

    def gen_cache(self, memory, do_zip=False):
        r"""
        Generates cache for `forward` usage. The generated cache is a list, and
        each element in it is a tuple( :code:`(incremental_cache, static_cache)` )
        produced by `TransformerDecoderLayer.gen_cache`. See `TransformerDecoderLayer.gen_cache`
        for more details. If `do_zip` is True, apply `zip` on these tuples to get
        a list with two elements.
       """
        cache = [layer.gen_cache(memory) for layer in self.layers]
        if do_zip:
            cache = list(zip(*cache))
        return cache


class TransformerDecoderLayer(nn.Layer):
    """
    The transformer decoder layer.

    It contains multiheadattention and some linear layers.
    """

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout=0.1,
                 activation="gelu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=True,
                 weight_attr=None,
                 bias_attr=None,
                 topo=None):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3

        super(TransformerDecoderLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        weight_attrs = _convert_param_attr_to_list(weight_attr, 3)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 3)

        self.self_attn = MultiHeadAttention(
            d_model,
            nhead,
            dropout=attn_dropout,
            weight_attr=weight_attrs[0],
            bias_attr=bias_attrs[0])

        if topo is None or topo.mp.size == 1:
            self.linear1 = nn.Linear(
                d_model,
                dim_feedforward,
                weight_attrs[2],
                bias_attr=bias_attrs[2])
            self.linear2 = nn.Linear(
                dim_feedforward,
                d_model,
                weight_attrs[2],
                bias_attr=bias_attrs[2])
        else:
            self.linear1 = ColumnParallelLiner(
                (d_model, dim_feedforward),
                topo.mp.size,
                weight_attrs[2],
                bias_attr=bias_attrs[2])
            self.linear2 = RowParallelLiner(
                (dim_feedforward, d_model),
                topo.mp.size,
                weight_attrs[2],
                bias_attr=bias_attrs[2])

        self.norm1 = nn.LayerNorm(d_model, epsilon=1e-5)
        self.norm2 = nn.LayerNorm(d_model, epsilon=1e-5)
        self.dropout1 = nn.Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = nn.Dropout(act_dropout, mode="upscale_in_train")
        self.activation = getattr(F, activation)

    def forward(self, tgt, memory, tgt_mask=None, use_cache=False, cache=None):
        residual = tgt

        if self.normalize_before:
            tgt = self.norm1(tgt)

        if use_cache is False:
            tgt = self.self_attn(tgt, tgt, tgt, tgt_mask, use_cache, cache)
        else:
            tgt, incremental_cache = self.self_attn(tgt, tgt, tgt, tgt_mask,
                                                    use_cache, cache)
        tgt = residual + self.dropout1(tgt)
        if not self.normalize_before:
            tgt = self.norm1(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm2(tgt)
        tgt = self.dropout2(
            self.linear2(F.gelu(
                self.linear1(tgt), approximate=True)))
        tgt = residual + tgt

        if not self.normalize_before:
            tgt = self.norm2(tgt)

        return tgt if use_cache is False else (tgt, incremental_cache)

    def gen_cache(self, memory):
        incremental_cache = self.self_attn.gen_cache(
            memory, type=self.self_attn.Cache)
        return incremental_cache


class GPT2Embeddings(nn.Layer):
    """
    Include embeddings from word, position and token_type embeddings
    """

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02,
                 topo=None):
        super(GPT2Embeddings, self).__init__()
        if topo is None or topo.mp.size == 1:
            self.word_embeddings = nn.Embedding(
                vocab_size,
                hidden_size,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(
                    mean=0.0, std=initializer_range)))
            self.position_embeddings = nn.Embedding(
                max_position_embeddings,
                hidden_size,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(
                    mean=0.0, std=initializer_range)))
        else:
            self.word_embeddings = ParallelEmbedding(
                vocab_size,
                hidden_size,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(
                    mean=0.0, std=initializer_range)))
            self.position_embeddings = ParallelEmbedding(
                max_position_embeddings,
                hidden_size,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(
                    mean=0.0, std=initializer_range)))

        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, position_ids=None):
        if position_ids is None:
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=1)
            position_ids = seq_length - ones
        input_embedings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = input_embedings + position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class GPT2PretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained GPT2 models. It provides GPT2 related
    `model_config_file`, `resource_files_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading and
    loading pretrained models. See `PretrainedModel` for more details.
    """

    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "gpt2-base-cn": {
            "vocab_size": 30000,
            "hidden_size": 2560,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "intermediate_size": 10240,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 1024,
            "type_vocab_size": 1,  # no use
            "initializer_range": 0.02,
            "pad_token_id": 0,
        },
        "gpt2-large-en": {
            "vocab_size": 50304,
            "hidden_size": 4096,
            "num_hidden_layers": 50,
            "num_attention_heads": 32,
            "intermediate_size": 16384,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 1024,
            "type_vocab_size": 1,  # no use
            "initializer_range": 0.02,
        },
        "gpt2-medium-en": {
            "vocab_size": 50304,
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "intermediate_size": 4096,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 1024,
            "type_vocab_size": 1,  # no use
            "initializer_range": 0.02,
        },
        "gpt2-small-en": {
            "vocab_size": 50304,
            "hidden_size": 1024,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "intermediate_size": 4096,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 1024,
            "type_vocab_size": 1,  # no use
            "initializer_range": 0.02,
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "gpt2-base-cn":
            "https://paddlenlp.bj.bcebos.com/models/transformers/gpt2/gpt2-base-cn.pdparams",
            "gpt2-medium-en":
            "https://paddlenlp.bj.bcebos.com/models/transformers/gpt2/gpt2-medium-en.pdparams",
        }
    }
    base_model_prefix = "gpt2"

    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.initializer_range
                        if hasattr(self, "initializer_range") else
                        self.gpt2.config["initializer_range"],
                        shape=layer.weight.shape))


@register_base_model
class GPT2Model(GPT2PretrainedModel):
    """
    The base model of gpt2.
    """

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02,
                 pad_token_id=0,
                 topo=None):
        super(GPT2Model, self).__init__()
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        self.topo = topo

        self.embeddings = GPT2Embeddings(
            vocab_size, hidden_size, hidden_dropout_prob,
            max_position_embeddings, type_vocab_size, self.initializer_range)

        decoder_layer = TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=intermediate_size,
            dropout=hidden_dropout_prob,
            activation=hidden_act,
            attn_dropout=attention_probs_dropout_prob,
            act_dropout=0,  # TODO @ZHUI check the dropout rate.
            weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(
                mean=0.0, std=self.initializer_range)),
            bias_attr=None,
            topo=topo)

        self.decoder = TransformerDecoder(
            decoder_layer, num_hidden_layers, norm=nn.LayerNorm(hidden_size))
        self.apply(self.init_weights)
        self.checkpoints = []

    def forward(self,
                input_ids,
                position_ids=None,
                attention_mask=None,
                use_cache=False,
                cache=None):
        self.checkpoints = []
        if attention_mask is None:
            length = paddle.shape(input_ids)[1]
            # attention_mask = paddle.tensor.triu(
            #     (paddle.ones(
            #         (length, length),
            #         dtype=self.embeddings.word_embeddings.weight.dtype) * -1e9),
            #     1)
            # use bool mask
            attention_mask = paddle.tensor.tril(
                paddle.ones(
                    (length, length),
                    dtype=self.embeddings.word_embeddings.weight.dtype))

        if position_ids is None:
            past_length = 0
            if cache is not None:
                past_length = paddle.shape(cache[0].k)[-2]
            position_ids = paddle.arange(
                past_length,
                paddle.shape(input_ids)[-1] + past_length,
                dtype='int64')
            position_ids = position_ids.unsqueeze(0)
            # .expand_as(input_ids)
            position_ids = paddle.fluid.layers.expand_as(position_ids,
                                                         input_ids)
        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids)
        # print_t("embedding_output", embedding_output)
        encoder_outputs = self.decoder(
            embedding_output,
            memory=None,
            tgt_mask=attention_mask,
            use_cache=use_cache,
            cache=cache)
        self.checkpoints.extend(self.decoder.checkpoints)
        return encoder_outputs


class GPT2ForPretraining(GPT2PretrainedModel):
    """
    The pretraining model of GPT2.

    It returns some logits and cached_kvs.
    """

    def __init__(self, gpt2):
        super(GPT2ForPretraining, self).__init__()
        self.gpt2 = gpt2
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                position_ids=None,
                attention_mask=None,
                masked_positions=None,
                use_cache=False,
                cache=None):
        outputs = self.gpt2(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            cache=cache)
        if use_cache:
            encoder_outputs, cached_kvs = outputs[:2]
        else:
            encoder_outputs = outputs
        logits = paddle.matmul(
            encoder_outputs,
            self.gpt2.embeddings.word_embeddings.weight,
            transpose_y=True)

        if use_cache:
            return logits, cached_kvs
        else:
            return logits


class GPT2PretrainingCriterion(paddle.nn.Layer):
    """
    Criterion for GPT2.

    It calculates the final loss.
    """

    def __init__(self):
        super(GPT2PretrainingCriterion, self).__init__()
        self.loss_func = paddle.nn.CrossEntropyLoss(reduction="none")

    def forward(self, prediction_scores, masked_lm_labels, loss_mask):
        masked_lm_loss = self.loss_func(prediction_scores,
                                        masked_lm_labels.unsqueeze(2))
        loss_mask = loss_mask.reshape([-1])
        masked_lm_loss = paddle.sum(masked_lm_loss.reshape([-1]) * loss_mask)
        loss = masked_lm_loss / loss_mask.sum()
        return loss


class GPT2ForGreedyGeneration(GPT2PretrainedModel):
    """
    The generate model for GPT-2.
    It use the greedy stategy and generate the next word with highest probablity.
    """

    def __init__(self, gpt2, max_predict_len):
        super(GPT2ForGreedyGeneration, self).__init__()
        self.gpt2 = gpt2
        self.max_predict_len = max_predict_len
        self.apply(self.init_weights)

    def model(self,
              input_ids,
              position_ids=None,
              attention_mask=None,
              masked_positions=None,
              use_cache=False,
              cache=None):
        outputs = self.gpt2(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            cache=cache)
        if use_cache:
            encoder_outputs, cached_kvs = outputs[:2]
        else:
            encoder_outputs = outputs
        logits = paddle.matmul(
            encoder_outputs,
            self.gpt2.embeddings.word_embeddings.weight,
            transpose_y=True)

        if use_cache:
            return logits, cached_kvs
        else:
            return logits

    def forward(self, input_ids, end_id):
        output, cached_kvs = self.model(input_ids, use_cache=True, cache=None)
        src_ids = input_ids
        nid = paddle.argmax(output[0, -1]).reshape([1, -1])
        src_ids = paddle.concat([src_ids, nid], axis=1)
        cur_len = 0
        # for i in range(max_predict_len):
        while (cur_len < self.max_predict_len):
            output, cached_kvs = self.model(
                nid, use_cache=True, cache=cached_kvs)

            nid = paddle.argmax(output[0, -1]).reshape([1, -1])
            src_ids = paddle.concat([src_ids, nid], axis=1)
            cur_len += 1
            if paddle.max(nid) == end_id:
                break

        return src_ids


class GPT2ForTopKPGeneration(GPT2PretrainedModel):
    """
    The generate model for GPT-2.
    It use the topk topk stategy to generation.
    """

    def __init__(self, gpt2, max_predict_len):
        super(GPT2ForTopKPGeneration, self).__init__()
        self.gpt2 = gpt2
        self.max_predict_len = max_predict_len
        self.apply(self.init_weights)

    def model(self,
              input_ids,
              position_ids=None,
              attention_mask=None,
              masked_positions=None,
              use_cache=False,
              cache=None):
        outputs = self.gpt2(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            cache=cache)
        if use_cache:
            encoder_outputs, cached_kvs = outputs[:2]
        else:
            encoder_outputs = outputs
        logits = paddle.matmul(
            encoder_outputs,
            self.gpt2.embeddings.word_embeddings.weight,
            transpose_y=True)

        if use_cache:
            return logits, cached_kvs
        else:
            return logits

    def forward(self, input_ids, end_id):
        output, cached_kvs = self.model(input_ids, use_cache=True, cache=None)
        src_ids = input_ids
        nid = paddle.argmax(output[0, -1]).reshape([1, -1])
        src_ids = paddle.concat([src_ids, nid], axis=1)
        cur_len = 0
        # for i in range(max_predict_len):
        while (cur_len < self.max_predict_len):
            output, cached_kvs = self.model(
                nid, use_cache=True, cache=cached_kvs)
            #TODO @zhui add topk topp support.
            #nid = paddle.argmax(output[0, -1]).reshape([1, -1])
            src_ids = paddle.concat([src_ids, nid], axis=1)
            cur_len += 1
            if paddle.max(nid) == end_id:
                break

        return src_ids
