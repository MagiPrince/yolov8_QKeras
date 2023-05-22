import tensorflow as tf
from keras.layers import *

class Conv(tf.keras.layers.Layer):
    def __init__(self, c_out, kernel=1, stride=1, padding="same", groups=1):
        super(Conv, self).__init__()

        self.conv = Conv2D(filters=c_out, kernel_size=kernel, strides=stride, padding=padding, groups=groups, use_bias=False)
        self.bn = BatchNormalization(momentum=0.95)
        self.relu = ReLU()

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        return self.relu(x)


class Bottleneck(tf.keras.layers.Layer):
    def __init__(self, c_out, shortcut=True, e=0.5, groups=1):
        super(Bottleneck, self).__init__()
        
        self.shortcut = shortcut
        self.c_out = c_out
        self.c = int(c_out * e)
        self.conv1 = Conv(c_out=self.c, kernel=3, stride=1)
        self.conv2 = Conv(c_out=c_out, kernel=3, stride=1, groups=groups)

    def forward(self, inputs):
        in_shape = tf.shape(inputs)
        if self.shortcut and in_shape[-1] == self.c_out:
            return inputs + self.conv2(self.conv1(inputs))
        else:
            return self.conv2(self.conv1(inputs))
        

class C2f(tf.keras.layers.Layer):
    def __init__(self, c_out, shortcut=False, n=1, e=0.5, groups=1):
        super(C2f, self).__init__()

        self.c = int(c_out * e)
        self.conv1 = Conv(c_out=c_out)
        self.conv2 = Conv(c_out=c_out)
        self.bottlenecks = [Bottleneck(c_out=self.c, shortcut=shortcut, groups=groups, e=1) for _ in range(n)]

    def forward(self, inputs):
        x = self.conv1(inputs)
        y = tf.split(x, num_or_size_splits=2, axis=-1)
        y.extend(bottleneck(y[-1]) for bottleneck in self.bottlenecks)
        return self.conv2(tf.keras.layers.Concatenate(axis=-1)(y))


class SPPF():
    def __init__(self, c_out):
        super(SPPF, self).__init__()

    def forward(self, inputs):
        return None
    

class Detect():
    def __init__(self, c_out, nc=1):
        super(Detect, self).__init__()

    def forward(self, inputs):
        return None