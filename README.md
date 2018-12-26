python3 --version
pip3 --version
virtualenv --version
si no estan:

sudo apt update
sudo apt install python3-dev python3-pip
sudo pip3 install -U virtualenv  # system-wide install

crear nuevo env:
virtualenv --system-site-packages -p python3 ./venv

activarlo
source ./venv/bin/activate  # sh, bash, ksh, or zsh

actualizar pip
pip install --upgrade pip

pip list  # show packages installed within the virtual environment

salir del env :
deactivate  # don't exit until you're done using TensorFlow

instalar en el ambiente
pip install --upgrade tensorflow

Recent tensorflow versions require the SoC to include AVX extensions which are not available on Cherry Trail and Apollo Lake based platforms (like UP and UP Squared).
pip3 install tensorflow==1.5
