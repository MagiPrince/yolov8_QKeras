import tensorflow as tf
from keras.layers import *
from common import *
from model import YoloV8

# t = tf.ones([1, 8, 8, 3])
# print(t)
# print(t.shape)

# t_temp = Conv(16, 3, 2)(t)
# t_temp = Conv(32, 3, 2)(t_temp)

# print(t_temp)
# print(t_temp.shape)
# t_temp = C2f(32, True, 1)(t_temp)
# print(t_temp)
# SPPF(32, 32, 5)(t_temp)
# t_final = Detect(32, 1)(t_temp)
# print(t_final)

t = tf.ones([1, 64, 64, 3])
print(YoloV8()(t))