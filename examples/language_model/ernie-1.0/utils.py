import paddle
import paddle.nn as nn
from paddle.nn import Layer
import paddle.nn.functional as F

old_init = paddle.nn.layer.common.Linear.__init__


def new_init(self, *k, **kw):
    old_init(self, *k, **kw)
    self.dropout = nn.Dropout(p=0.1)


def new_forward(self, input):
    out = F.linear(
        x=input,
        weight=self.dropout(self.weight),
        bias=self.bias,
        name=self.name)
    return out


paddle.nn.layer.common.Linear.__init__ = new_init
paddle.nn.layer.common.Linear.forward = new_forward
