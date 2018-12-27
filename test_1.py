import tensorflow as tf
import numpy as np

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

x=tf.Variable(3,name="x")
y=tf.Variable(4,name="y")
f=x*x*y+y+2
#sin sesion
# sess=tf.Session()
# sess.run(x.initializer)
# sess.run(y.initializer)
# result=sess.run(f)
# print(result)
# sess.close()

#con sesion
with tf.Session() as sess:
 x.initializer.run()
 y.initializer.run()
 result = f.eval()
 print(result)


# Instead of manually running the initializer for every single variable, you can use the
# global_variables_initializer()  function. Note that it does not actually perform the initialization
# immediately, but rather creates a node in the graph that will initialize all variables when it is run:
init = tf.global_variables_initializer()  
# prepare an init node
with tf.Session() as sess:
 init.run()  
 # actually initialize all the variables
 result = f.eval()
 print(result,"asd")

#cualquier nodo creado es agregado en automatico al grafo x default
x1 = tf.Variable(1)
print(x1.graph is tf.get_default_graph())

#si queremos mantenerlos separados usar bucle:


graph = tf.Graph()
with graph.as_default():
 x2 = tf.Variable(2)

 print(x2.graph is graph,
  x2.graph is tf.get_default_graph())

#aunque se repita codigo lo evalua individual
w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3
with tf.Session() as sess:
 print(y.eval())  # 10
 print(z.eval()) 

#hay que indicarle que los corra juntos ya horre calculos
with tf.Session() as sess:
 y_val, z_val = sess.run([y, z])
 print(y_val)  # 10
 print(z_val)  # 15

reset_graph()
# se puede usar como numpy para acceder al GPU==============

import numpy as np
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
# X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
# y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
# XT = tf.transpose(X)
# theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)
# with tf.Session() as sess:
#   theta_value = theta.eval()
#   print(theta_value)    

# #=usando las funciones gradientes de tf=====================================
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

# reset_graph()
# n_epochs	=	1000
# learning_rate	=	0.01
# X	=	tf.constant(scaled_housing_data_plus_bias,	dtype=tf.float32,	name="X")
# y	=	tf.constant(housing.target.reshape(-1,	1),	dtype=tf.float32,	name="y")
# theta	=	tf.Variable(tf.random_uniform([n	+	1,	1],	-1.0,	1.0),	name="theta")
# y_pred	=	tf.matmul(X,	theta,	name="predictions")
# error	=	y_pred	-	y
# mse	=	tf.reduce_mean(tf.square(error),	name="mse")
# # gradients	=	2/m	*	tf.matmul(tf.transpose(X),	error)
# # gradients = tf.gradients(mse, [theta])[0]
# # training_op	=	tf.assign(theta,	theta	-	learning_rate	*	gradients)
# optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
# training_op = optimizer.minimize(mse)
# init	=	tf.global_variables_initializer()
# with	tf.Session()	as	sess:
# 	sess.run(init)
# 	for	epoch	in	range(n_epochs):
# 		if	epoch	%	100	==	0:
# 			print("Epoch",	epoch,	"MSE	=",	mse.eval())
# 		sess.run(training_op)
# 	best_theta	=	theta.eval()
# print(best_theta)
# # grafos para alimentar de datos
# reset_graph()
# A = tf.placeholder(tf.float32, shape=(None, 3))
# B = A + 5
# with tf.Session() as sess:
#     B_val_1 = B.eval(feed_dict={A: [[1, 2, 3]]})
#     B_val_2 = B.eval(feed_dict={A: [[4, 5, 6], [7, 8, 9]]})
# print(B_val_1,B_val_2)

# #con placeholder:=========================================

# reset_graph()
# n_epochs = 10
# batch_size = 100
# n_batches = int(np.ceil(m / batch_size))
# #parte los datos
def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
    indices = np.random.randint(m, size=batch_size)  # not shown
    X_batch = scaled_housing_data_plus_bias[indices] # not shown
    y_batch = housing.target.reshape(-1, 1)[indices] # not shown
    return X_batch, y_batch

# #establece los placeholders, lo demas igual
# X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
# y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
# theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
# y_pred = tf.matmul(X, theta, name="predictions")
# error = y_pred - y
# mse = tf.reduce_mean(tf.square(error), name="mse")
# #en lugar de usar momento usamos gradiente
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# training_op = optimizer.minimize(mse)
# init = tf.global_variables_initializer()

# with tf.Session() as sess:
# 	sess.run(init)

# 	for epoch in range(n_epochs):
# 		#para cada batch
# 		for batch_index in range(n_batches):
# 			#generar los batches
# 			X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
# 			#alimentar al minimizador con los datos
# 			sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

# 			best_theta = theta.eval()
# print(best_theta)

#almacenando=============================
# reset_graph()

# n_epochs = 1000                                                                       # not shown in the book
# learning_rate = 0.01                                                                  # not shown

# X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")            # not shown
# y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")            # not shown
# #solo se agrego theta
# theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
# y_pred = tf.matmul(X, theta, name="predictions")                                      # not shown
# error = y_pred - y                                                                    # not shown
# mse = tf.reduce_mean(tf.square(error), name="mse")                                    # not shown
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)            # not shown
# training_op = optimizer.minimize(mse)                                                 # not shown
# #lo de abajo si se modifico del anterior
# init = tf.global_variables_initializer()
# saver = tf.train.Saver()
# ##
# saver = tf.train.import_meta_graph("/tmp/my_model_final.ckpt.meta")  # this loads the graph structure
# # theta = tf.get_default_graph().get_tensor_by_name("theta:0") # not shown in the book

# #recuperar
# with tf.Session() as sess:
#   saver.restore(sess, "/tmp/my_model_final.ckpt")
#   best_theta_restored = theta.eval() # not shown in the book
#   # print(best_theta_restored)

#   sess.run(init)
#   for epoch in range(n_epochs):
#     if epoch % 100 == 0:
#       print("Epoch", epoch, "MSE =", mse.eval())                                # not shown
#       save_path = saver.save(sess, "/tmp/my_model.ckpt")
#     sess.run(training_op)
#   best_theta = theta.eval()
#   print(best_theta)
# 	#para salcar solo una variable
# 	# saver = tf.train.Saver({"weights": theta})

#usando tesorboard============================
from	datetime	import	datetime
now	=	datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir	=	"tf_logs"
logdir	=	"{}/run-{}/".format(root_logdir,	now)

reset_graph() 
batch_size = 10000
n_batches = int(np.ceil(m / batch_size))
print(n_batches)
n_epochs = 1000
learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
with	tf.name_scope("loss")	as	scope:
	error	=	y_pred	-	y
	mse	=	tf.reduce_mean(tf.square(error),	name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

#recuperar
with tf.Session() as sess:
  sess.run(init)
  for epoch in range(n_epochs):

    for batch_index in range(n_batches):
      X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
      if batch_index % 100 == 0:
        summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
        step = epoch * n_batches + batch_index
        file_writer.add_summary(summary_str, step)
        print("step", step)
      sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

  best_theta = theta.eval()
  file_writer.close()
print(best_theta)

         
