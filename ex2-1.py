#Chapter 2 - Math in TensorFlow
#One Dimensional Data Structure

import numpy as np
import tensorflow as tf
#tensor_1d = np.array([1.3, 1, 4.0, 23.99], [2.0, 3.0, 4.0])
tensor_1d = np.array([1.3, 1, 4.0, 23.99])
print(tensor_1d)
print(tensor_1d.ndim)
print(tensor_1d.shape)
print(tensor_1d.dtype)
tf_tensor = tf.convert_to_tensor(tensor_1d, dtype=tf.float64)
sess = tf.Session()
print(sess.run(tf_tensor))
print(sess.run(tf_tensor[0]))
print(sess.run(tf_tensor[2]))