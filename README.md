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
