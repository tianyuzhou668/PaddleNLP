# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

# This file is autogenerated by the command `make fix-copies`, do not edit.
from . import DummyObject, requires_backends


class AutoencoderKL(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])


class ControlNetModel(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])


class ModelMixin(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])


class PriorTransformer(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])


class Transformer2DModel(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])


class UNet1DModel(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])


class UNet2DConditionModel(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])


class UNet2DModel(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])


class UNet3DConditionModel(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])


class VQModel(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])


def get_constant_schedule(*args, **kwargs):
    requires_backends(get_constant_schedule, ["paddle"])


def get_constant_schedule_with_warmup(*args, **kwargs):
    requires_backends(get_constant_schedule_with_warmup, ["paddle"])


def get_cosine_schedule_with_warmup(*args, **kwargs):
    requires_backends(get_cosine_schedule_with_warmup, ["paddle"])


def get_cosine_with_hard_restarts_schedule_with_warmup(*args, **kwargs):
    requires_backends(get_cosine_with_hard_restarts_schedule_with_warmup, ["paddle"])


def get_linear_schedule_with_warmup(*args, **kwargs):
    requires_backends(get_linear_schedule_with_warmup, ["paddle"])


def get_polynomial_decay_schedule_with_warmup(*args, **kwargs):
    requires_backends(get_polynomial_decay_schedule_with_warmup, ["paddle"])


def get_scheduler(*args, **kwargs):
    requires_backends(get_scheduler, ["paddle"])


class AudioPipelineOutput(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])


class DanceDiffusionPipeline(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])


class DDIMPipeline(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])


class DDPMPipeline(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])


class DiffusionPipeline(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])


class DiTPipeline(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])


class ImagePipelineOutput(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])


class KarrasVePipeline(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])


class LDMPipeline(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])


class LDMSuperResolutionPipeline(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])


class PNDMPipeline(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])


class RePaintPipeline(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])


class ScoreSdeVePipeline(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])


class DDIMInverseScheduler(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])


class DDIMScheduler(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])


class DDPMScheduler(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])


class DEISMultistepScheduler(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])


class DPMSolverMultistepScheduler(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])


class DPMSolverSinglestepScheduler(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])


class EulerAncestralDiscreteScheduler(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])


class PreconfigEulerAncestralDiscreteScheduler(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])


class EulerDiscreteScheduler(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])


class HeunDiscreteScheduler(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])


class IPNDMScheduler(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])


class KarrasVeScheduler(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])


class KDPM2AncestralDiscreteScheduler(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])


class KDPM2DiscreteScheduler(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])


class PNDMScheduler(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])


class RePaintScheduler(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])


class SchedulerMixin(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])


class ScoreSdeVeScheduler(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])


class UnCLIPScheduler(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])


class UniPCMultistepScheduler(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])


class VQDiffusionScheduler(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])


class EMAModel(metaclass=DummyObject):
    _backends = ["paddle"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["paddle"])

    @classmethod
    def from_config(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["paddle"])
