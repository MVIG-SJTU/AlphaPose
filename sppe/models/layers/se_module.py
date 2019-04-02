import mxnet.gluon.nn as nn
from mxnet.gluon.block import HybridBlock


class SELayer(HybridBlock):
    def __init__(self, channel, reduction=1):
        super(SELayer, self).__init__()
        with self.name_scope():
            self.fc = nn.HybridSequential()
            self.fc.add(nn.Dense(channel // reduction))
            self.fc.add(nn.Activation('relu'))
            self.fc.add(nn.Dense(channel, activation='sigmoid'))

    def hybrid_forward(self, F, x):
        y = F.contrib.AdaptiveAvgPooling2D(x, output_size=1)
        y = self.fc(y)

        return y.expand_dims(-1).expand_dims(-1).broadcast_like(x) * x
