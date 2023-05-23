import tensorflow as tf
from keras.layers import *

class Conv(tf.keras.layers.Layer):
    def __init__(self, c_out, kernel_size=1, stride=1, padding="same", dilation_rate=1, groups=1):
        super(Conv, self).__init__()

        self.conv = Conv2D(filters=c_out, kernel_size=kernel_size, strides=stride, padding=padding, dilation_rate=1, groups=groups, use_bias=False)
        self.bn = BatchNormalization(momentum=0.99)
        self.relu = ReLU()

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        return self.relu(x)


class Bottleneck(tf.keras.layers.Layer):
    def __init__(self, c_out, shortcut=True, e=0.5, groups=1):
        super(Bottleneck, self).__init__()
        
        self.shortcut = shortcut
        self.c_out = c_out
        self.c = int(c_out * e)
        self.conv1 = Conv(c_out=self.c, kernel_size=3, stride=1)
        self.conv2 = Conv(c_out=c_out, kernel_size=3, stride=1, groups=groups)

    def call(self, inputs):
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

    def call(self, inputs):
        x = self.conv1(inputs)
        y = tf.split(x, num_or_size_splits=2, axis=-1)
        y.extend(bottleneck(y[-1]) for bottleneck in self.bottlenecks)
        return self.conv2(Concatenate(axis=-1)(y))


class SPPF():
    def __init__(self, c_in, c_out, kernel_size=5):
        super(SPPF, self).__init__()

        c_ = c_in // 2
        self.conv1 = Conv(c_out=c_)
        self.conv2 = Conv(c_out=c_out)
        self.maxpool = MaxPooling2D(pool_size=(kernel_size, kernel_size), strides=1, padding="same")

    def call(self, inputs):
        x = self.conv1(inputs)
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)
        concat = Concatenate(axis=-1)([x, y1, y2, y3])
        return self.conv2(concat)
    

class Detect():
    def __init__(self, c_out, nc=1):
        super(Detect, self).__init__()

        self.nc = nc  # number of classes
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)

        self.conv1 = Conv(c_out=c_out, kernel_size=3, stride=1)
        self.conv2 = Conv(c_out=c_out, kernel_size=3, stride=1)

    def call(self, inputs):
        return None