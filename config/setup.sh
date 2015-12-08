echo "preparing variable senviroments"
export PATH=/home/ubuntu/anaconda/bin:${PATH}

echo "updatting the system "
sudo apt-get update ## && sudo apt-get upgrade -y
echo "installing packages "
echo "\tinstalling git"
sudo apt-get -y install git
echo "\tinstalling make"
sudo apt-get -y install make
echo "\tinstalling htop"
sudo apt-get -y install htop
echo "\tinstalling g++"
sudo apt-get -y install g++

echo "updatting pip"
pip install --upgrade pip
echo "installing nose"
pip install nose

echo "clone xgboost"
git clone https://github.com/dmlc/xgboost.git
echo "building xgboost"
cd xgboost
./build.sh
echo "python setting up"
cd python-package
python setup.py install
