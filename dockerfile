FROM chaneyk/pytorch:1.3.0-py3

RUN pip3 install -U pip
RUN pip3 install --no-deps pyrender smplx ipdb

WORKDIR /opt
RUN wget https://spdf.gsfc.nasa.gov/pub/software/cdf/dist/cdf37_1/cdf-dist-all.tar.gz && gunzip cdf-dist-all.tar.gz && tar -xvf cdf-dist-all.tar
# RUN wget https://spdf.gsfc.nasa.gov/pub/software/cdf/dist/cdf37_0/cdf-dist-all.tar.gz && gunzip cdf-dist-all.tar.gz && tar -xvf cdf-dist-all.tar
WORKDIR cdf37_1-dist
RUN make OS=linux ENV=gnu CURSES=no all && make INSTALLDIR=/usr/local/cdf install && make clean
WORKDIR /opt
RUN git clone https://github.com/spacepy/spacepy.git
WORKDIR spacepy
RUN python3 setup.py install
RUN pip3 install tensorboard future
RUN pip3 install freetype-py pyglet

WORKDIR /
RUN git clone -b version01 https://github.com/Hamoon1987/SPINH.git
WORKDIR /SPINH
RUN pip3 install -r requirements.txt
RUN ./fetch_data.sh
# RUN python3 demo.py --checkpoint=data/model_checkpoint.pt --img=examples/im1010.jpg --openpose=examples/im1010_openpose.json