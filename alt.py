import tensorflow as tf
from time import time

beg = time()
a = tf.Variable(-10, name='a', dtype=tf.float32)
b = tf.Variable(10, name='b', dtype=tf.float32)

def g(x):
    return tf.clip_by_value( (x-a)/(b-a), 0, 1)

X = tf.lin_space(-20., 20., 2000)
loss = tf.reduce_sum( tf.square( tf.math.sigmoid(X) - g(X)))
opt = tf.train.AdamOptimizer(learning_rate=1e-3)
train_op = opt.minimize( loss)
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)

    for _ in range( int(1e4)):
        sess.run( train_op)

print( 'Non-eager run in %.1f seconds' %(time()-beg))