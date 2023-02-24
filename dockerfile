FROM chaneyk/pytorch:1.3.0-py3

RUN pip3 install -U pip
RUN pip3 install --no-deps smplx ipdb

WORKDIR /opt
RUN wget https://spdf.gsfc.nasa.gov/pub/software/cdf/dist/cdf37_1/cdf-dist-all.tar.gz && gunzip cdf-dist-all.tar.gz && tar -xvf cdf-dist-all.tar
WORKDIR cdf37_1-dist
RUN make OS=linux ENV=gnu CURSES=no all && make INSTALLDIR=/usr/local/cdf install && make clean
WORKDIR /opt
RUN git clone https://github.com/spacepy/spacepy.git
WORKDIR spacepy
RUN python3 setup.py install
RUN pip3 install tensorboard future
RUN pip3 install freetype-py pyglet==1.5.27

WORKDIR /
RUN git clone -b base https://github.com/Hamoon1987/meshConfidence
WORKDIR /meshConfidence
RUN pip3 install -r requirements.txt
RUN git clone https://github.com/Hzzone/pytorch-openpose pytorchopenpose
RUN   sed -i "s|from src import util|from pytorchopenpose.src import util |g" pytorchopenpose/src/body.py
RUN   sed -i "s|from src.model import bodypose_model|from pytorchopenpose.src.model import bodypose_model |g" pytorchopenpose/src/body.py
RUN ./fetch_data.sh