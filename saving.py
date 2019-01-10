import tensorflow as tf 
import numpy as np
import io, base64, os

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# x=tf.Variable(3,name="x")
# y=tf.Variable(4,name="y")
# f=x+y+2
res=tf.Variable(1,name="resultado")
x = tf.placeholder(tf.float32, None)
f=x+1

save_file = os.path.join(os.getcwd(), 'asd.ckpt')

saver = tf.train.Saver()

print("res es ",res)
with tf.Session() as sess:
	saver.restore(sess, "asd.ckpt")
	print("el saver es",res.eval())
	sess.run(tf.global_variables_initializer()) 

	#result =sess.run(f)
	result=f.eval(feed_dict={x:[1]})
	print("result es: ",result[0])

	save_path = saver.save(sess, save_file)
	print("Model saved in file: ", save_path)


# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# # Training steps
# saver = tf.train.Saver()
# with tf.Session() as sess:
# 	sess.run(tf.global_variables_initializer())
# 	saver.restore(sess, tensorflow_ckpt_file)
# 	classification = sess.run(tf.argmax(y, 1), feed_dict={x: [img], keep_prob: 1.0})
# 	return(classification[0])


# training_op = optimizer.minimize(mse)

# init = tf.global_variables_initializer()
# mse_summary = tf.summary.scalar('MSE', mse)
# file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

# #recuperar
# with tf.Session() as sess:
#   sess.run(init)
#   for epoch in range(n_epochs):

#     for batch_index in range(n_batches):
#       X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
#       if batch_index % 100 == 0:
#         summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
#         step = epoch * n_batches + batch_index
#         file_writer.add_summary(summary_str, step)
#         print("step", step)
#       sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

#   best_theta = theta.eval()
#   file_writer.close()
# print(best_theta)


















