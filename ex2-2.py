#ex2-2.py
#Chapter 2 - Math in TensorFlow
#Two Dimensional Tensor

import numpy as np
import tensorflow as tf

tensor_2d = np.array([(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12)])
print(tensor_2d) 
print(tensor_2d.ndim)#2, row & column
print(tensor_2d.shape)#3, 4
print(tensor_2d.dtype)

matrix1 = np.array([(2,2,2), (2,2,2), (2,2,2)], dtype='int32')
matrix2 = np.array([(3,3,3),(3,3,3),(3,3,3)], dtype='int32')
matrix3 = np.array([(2,7,2),(1,4,2),(9,0,2)], dtype='float32')
matrix1 = tf.constant(matrix1)
matrix2 = tf.constant(matrix2)
matrix_prod = tf.matmul(matrix1, matrix2)
matrix_sum = tf.add(matrix1, matrix2)
matrix_det = tf.matrix_determinant(matrix3)
sess = tf.Session()
print(sess.run(matrix_prod))
print(sess.run(matrix_sum))
print(sess.run(matrix_det))