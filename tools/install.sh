# exit on error
set -e

# create env and install pytorch
conda create -y -n TS-Diff python=3.8
conda activate TS-Diff
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# install packages
pip install -r requirements.txt
python setup.py develop

### install rawpy and LibRaw for benchmark
mkdir -p downloads/
# download LibRaw and rawpy
wget https://www.libraw.org/data/LibRaw-0.21.1.zip -O downloads/LibRaw-0.21.1.zip
python scripts/download_gdrive.py --id 1EuJsbZ_a_YJHHcGAVA9TXXPnGU90QoP4 --save-path downloads/rawpy.zip

unzip downloads/rawpy.zip -d downloads/
unzip downloads/LibRaw-0.21.1.zip -d downloads/

# setting LibRAW
cd downloads/LibRaw-0.21.1
./configure
# 如果没有sudo权限
# ./configure --prefix=/home/ly/.conda/envs/RawDiff/
#export PKG_CONFIG_PATH=/home/ly/.conda/envs/RawDiff/lib/pkgconfig
make
sudo make install

# setting rawpy
cd ../rawpy
RAWPY_USE_SYSTEM_LIBRAW=1 pip install -e .
