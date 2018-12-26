import tensorflow as tf
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
with tf.Session()	as sess:
	x.initializer.run()
	y.initializer.run()
	result	=	f.eval()
	print(result)


# Instead of manually running the initializer for every single variable,	you can use the
# global_variables_initializer() 	function.	Note that it does not actually perform the initialization
# immediately,	but rather creates a node in the graph that will initialize all variables when it is run:
init	=	tf.global_variables_initializer()		
#	prepare an init node
with tf.Session()	as sess:
	init.run()		
	#	actually initialize all the variables
	result =	f.eval()
	print(result,"asd")

#cualquier nodo creado es agregado en automatico al grafo x default
x1	=	tf.Variable(1)
print(x1.graph is tf.get_default_graph())

#si queremos mantenerlos separados usar bucle:


graph	=	tf.Graph()
with graph.as_default():
	x2	=	tf.Variable(2)

	print(x2.graph is graph,
		x2.graph is tf.get_default_graph())

#aunque se repita codigo lo evalua individual
w	=	tf.constant(3)
x	=	w	+	2
y	=	x	+	5
z	=	x	*	3
with tf.Session()	as sess:
	print(y.eval())		#	10
	print(z.eval())	

#hay que indicarle que los corra juntos ya horre calculos
with tf.Session()	as sess:
	y_val,	z_val	=	sess.run([y,	z])
	print(y_val)		#	10
	print(z_val)		#	15

#se puede usar como numpy para acceder al GPU
# import numpy as np
# from sklearn.datasets import fetch_california_housing
# housing	=	fetch_california_housing()
# m,	n	=	housing.data.shape
# housing_data_plus_bias	=	np.c_[np.ones((m,	1)),	housing.data]
# X	=	tf.constant(housing_data_plus_bias,	dtype=tf.float32,	name="X")
# y	=	tf.constant(housing.target.reshape(-1,	1),	dtype=tf.float32,	name="y")
# XT	=	tf.transpose(X)
# theta	=	tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT,	X)),	XT),	y)
# with tf.Session()	as sess:
# 				theta_value	=	theta.eval()

#falta hacer el gradiente descendente