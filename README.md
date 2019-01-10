Deep Learning
===

Dependencias:

	python3 --version
	pip3 --version
	virtualenv --version
si no estan:

	sudo apt update
	sudo apt install python3-dev python3-pip
	sudo pip3 install -U virtualenv  # system-wide install

crear nuevo env:

	virtualenv --system-site-packages -p python3 ./venv
	source ./venv/bin/activate  # sh, bash, ksh, or zsh
	pip install --upgrade pip
	pip list  # show packages installed within the virtual environment

salir del env :

	deactivate  # don't exit until you're done using TensorFlow

instalar en el ambiente:

	pip install --upgrade tensorflow
	Recent tensorflow versions require the SoC to include AVX extensions
	pip install tensorflow==1.5


Jupyter -> InteractiveSession is	created	it	automatically	setsitself	as	the	default	session,	so	you	don’t	need	a	 with 	block	(but	you	do	need	to	close	the	session
manually):

	sess	=	tf.InteractiveSession()
	init.run()
	result	=	f.eval()
	print(result)
	sess.close()



los datos a grabar:

	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
	se agrega el valor a grabar 
	acc = tf.summary.scalar('acc', accuracy)
	file_writer = tf.summary.FileWriter("tf_logs/percep", tf.get_default_graph())
	se evalua
	acc_val = acc.eval(feed_dict={X: X_valid, y: y_valid})
	se graba el archivo
	file_writer.add_summary(acc_val,epoch)


hay 2 etapas de ejecucion, una de entrenamiento y ahi almacenaremos los datos entrenados que seran leiso por una en tiempo de ejecucion que solo evaluara el dato ingresado
la red se tieen que construir igual en produccion y en entrenamiento:

training: 

	with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    max_steps = 2000
    for step in range(max_steps):
        batch_xs, batch_ys = mnist.train.next_batch(50)
        if (step % 100) == 0:
            print(step, sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
    print(max_steps, sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

    saver = tf.train.Saver()
    save_path = saver.save(sess, save_file)
    print ("Model saved in file: ", save_path)
 

 prod:

	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		saver.restore(sess, tensorflow_ckpt_file)
		classification = sess.run(tf.argmax(y, 1), feed_dict={x: [img], keep_prob: 1.0})
		return(classification[0])
 


