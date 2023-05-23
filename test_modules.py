import tensorflow as tf
from keras.layers import *
from common import *

t = tf.ones([1, 8, 8, 3])
print(t)
print(t.shape)

t_temp = Conv(16, 3, 2)(t)
t_temp = Conv(32, 3, 2)(t_temp)

print(t_temp)
print(t_temp.shape)
t_temp = C2f(32, True, 1)(t_temp)
print(t_temp)