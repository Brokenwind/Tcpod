import sys
import tensorflow as tf
from read_data import *

def main():
  # Import data
  batch_size = 512
  train_imgs, train_labels, test_imgs, test_labels = load_all()
  m,n = train_imgs.shape

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([1, 10]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])
  #cross_entropy = -tf.reduce_sum(y_*tf.log(y))
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

  sess = tf.Session()
  init = tf.global_variables_initializer()
  sess.run(init)
  # Train
  for i in range(10):
    batches = gen_batches(m, batch_size)
    for batch in batches:
      batch_xs = train_imgs[batch]
      batch_ys = train_labels[batch]
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  print(sess.run(W))
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: train_imgs, y_: train_labels}))

if __name__ == '__main__':
  main()
