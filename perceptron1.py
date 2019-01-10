
# Common imports
import numpy as np
import os
import tensorflow as tf

from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


n_inputs = 28*28  # MNIST
n_hidden1 = 100
n_hidden2 = 80
n_outputs = 10

reset_graph()

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:4000], X_train[4000:]
y_valid, y_train = y_train[:4000], y_train[4000:]


X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

def neuron_layer(X, n_neurons, name, activation=None):
 with tf.name_scope(name):
  n_inputs = int(X.get_shape()[1])
  stddev = 2 / np.sqrt(n_inputs)
  init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
  W = tf.Variable(init, name="kernel")
  b = tf.Variable(tf.zeros([n_neurons]), name="bias")
  Z = tf.matmul(X, W) + b
  if activation is not None:
   return activation(Z)
  else:
   return Z

with tf.name_scope("dnn"):
	hidden1 = neuron_layer(X, n_hidden1, name="hidden1", activation=tf.nn.relu)
	hidden2 = neuron_layer(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
	logits = neuron_layer(hidden2, n_outputs, name="outputs")

with tf.name_scope("loss"):
  xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
  loss = tf.reduce_mean(xentropy, name="loss")

learning_rate = 0.01

with tf.name_scope("train"):
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
  correct = tf.nn.in_top_k(logits, y, 1)
  accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
  acc = tf.summary.scalar('acc', accuracy)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 40
batch_size = 80

def shuffle_batch(X, y, batch_size):
  rnd_idx = np.random.permutation(len(X))
  n_batches = len(X) // batch_size
  for batch_idx in np.array_split(rnd_idx, n_batches):
    X_batch, y_batch = X[batch_idx], y[batch_idx]
    yield X_batch, y_batch

file_writer = tf.summary.FileWriter("tf_logs/percep", tf.get_default_graph())


with tf.Session() as sess:
  init.run()
  for epoch in range(n_epochs):

    for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
      sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    if epoch % 4 == 0:
     acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
     acc_val = acc.eval(feed_dict={X: X_valid, y: y_valid})
     file_writer.add_summary(acc_val,epoch)
     print(epoch, "Batch accuracy:", acc_batch, "Val accuracy:",acc_val)

  save_path = saver.save(sess, "tf_logs/percep/my_model_final.ckpt")

with tf.Session() as sess:
  saver.restore(sess, "tf_logs/percep/my_model_final.ckpt") # or better, use save_path
  X_new_scaled = X_test[:20]
  Z = logits.eval(feed_dict={X: X_new_scaled})
  y_pred = np.argmax(Z, axis=1)


print("Predicted classes:", y_pred)
print("Actual classes:   ", y_test[:20])

file_writer.close()

