import tensorflow as tf

"""
Once the code runs, we can load the data into TensorBoard via the command line:
  tensorboard --logdir /tmp/example_hist
"""
def main():
    k = tf.placeholder(tf.float32)
    mean_moving_normal = tf.random_normal(shape=[1000], mean=(5*k), stddev=1)
    tf.summary.histogram('normal/moving_mean', mean_moving_normal)
    sess = tf.Session()
    writer = tf.summary.FileWriter('/tmp/example_hist')
    summaries = tf.summary.merge_all()
    N = 400
    for step in range(N):
        k_val = step/float(N)
        summ = sess.run(summaries, feed_dict={k: k_val})
        writer.add_summary(summ, global_step = step)

if __name__ == '__main__':
    main()
