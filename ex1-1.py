x = 1
y = x + 9
print(y)
import tensorflow as tf
x = tf.constant(1, name='x')
y = tf.Variable(x+9, name='y')
#print(y)
model = tf.initialize_all_variables()
sess = tf.Session()
#merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter('/tmp/tensorflowlogs', sess.graph)
sess.run(model)
print(sess.run(y))
print(y)