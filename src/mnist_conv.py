import sys
import tensorflow as tf
from read_data import *

def weight_variable(shape):
  init = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(init)

def bias_variable(shape):
  init = tf.constant(0.1, shape=shape)
  return tf.Variable(init)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def main():
  # Import data
  batch_size = 512
  train_imgs, train_labels, test_imgs, test_labels = load_all()
  m,n = train_imgs.shape

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  y_ = tf.placeholder(tf.float32, [None, 10])
  # the first conv layer
  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  x_image = tf.reshape(x, [-1, 28, 28, 1])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)
  # the scecond conv layer
  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)
  # the full connected layer
  W_fc1 = weight_variable([7 * 7 * 64, 1024])
  b_fc1 = bias_variable([1024])
  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
  # dropout
  keep_prob = tf.placeholder("float")
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
  # output layer
  W_fc2 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])
  y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

  #ross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
  cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  for i in range(10):
    batches = gen_batches(m, batch_size)
    for batch in batches:
      batch_xs = train_imgs[batch]
      batch_ys = train_labels[batch]
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.6})
    print ( "Test accuracy in %dth iteration is %g" % (i, sess.run(accuracy,feed_dict={
      x: test_imgs[0:1000],
      y_: test_labels[0:1000],
      keep_prob: 1.0 })))

if __name__ == '__main__':
  main()
