source bash/init.sh
pip install gym==0.25.2
mkdir ..
pushd ..
git clone https://github.com/Farama-Foundation/Minigrid.git
cd Minigrid/
git checkout ac65e0312a37dc9c0fa0fc078dac947ecb075464
pip install -e .
popd
