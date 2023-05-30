import tensorflow as tf
from keras.layers import *
from common import *

model_d = {
    "n": 0.33,
    "s": 0.33,
    "m": 0.67,
    "l": 1.00,
    "x": 1.00
}

model_w = {
    "n": 0.25,
    "s": 0.50,
    "m": 0.75,
    "l": 1.00,
    "x": 1.25
}

model_r = {
    "n": 2.0,
    "s": 2.0,
    "m": 1.5,
    "l": 1.0,
    "x": 1.0
}

class YoloV8(tf.keras.Model):
    def __init__(self, model_size="s", number_of_classes=1):
        super().__init__()
        self.d = model_d[model_size]
        self.w = model_w[model_size]
        self.r = model_r[model_size]
        self.nc = number_of_classes

    def call(self, inputs):
        x = Conv(64*self.w, 3, 2)(inputs)
        x = Conv(128*self.w, 3, 2)(x)
        x = C2f(128*self.w, True, 3*self.d)(x)
        x = Conv(256*self.w, 3, 2)(x)
        x_4 = C2f(256*self.w, True, 6*self.d)(x)
        x = Conv(512*self.w, 3, 2)(x_4)
        x_6 = C2f(512*self.w, True, 6*self.d)(x)
        x = Conv(512*self.w*self.r, 3, 2)(x_6)
        x = C2f(512*self.w*self.r, True, 3*self.d)(x)
        x_9 = SPPF(512*self.w*self.r, 512*self.w*self.r)(x)
        x = UpSampling2D((2,2))(x_9)
        x = Concatenate(axis=-1)([x_6, x])
        x_12 = C2f(512*self.w, False, 3*self.d)(x)
        x = UpSampling2D((2,2))(x_12)
        x = Concatenate(axis=-1)([x_4, x])
        x_15 = C2f(256*self.w, False, 3*self.d)(x)
        x = Conv(256*self.w, 3, 2)(x_15)
        x = Concatenate(axis=-1)([x_12, x])
        x_18 = C2f(512*self.w, False, 3*self.d)(x)
        x = Conv(512*self.w, 3, 2)(x_18)
        x = Concatenate(axis=-1)([x_9, x])
        x_21 = C2f(512*self.w*self.r, False, 3*self.d)(x)

        y_1 = Detect(256*self.w, self.nc)(x_15)
        y_2 = Detect(512*self.w, self.nc)(x_18)
        y_3 = Detect(512*self.w*self.r, self.nc)(x_21)

        return y_1, y_2, y_3
