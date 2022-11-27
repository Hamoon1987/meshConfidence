# SPINH
1- Go to server creat a folder SPINH
2- Create a dockerfile:

	FROM chaneyk/pytorch:1.3.0-py3
	RUN pip3 install -U pip
	RUN pip3 install --no-deps pyrender smplx ipdb
	WORKDIR /opt
	RUN wget https://spdf.gsfc.nasa.gov/pub/software/cdf/dist/cdf37_1/cdf-dist-all.tar.gz && gunzip cdf-dist-all.tar.gz && tar -xvf cdf-dist-all.tar
	WORKDIR cdf37_1-dist
	RUN make OS=linux ENV=gnu CURSES=no all && make INSTALLDIR=/usr/local/cdf install && make clean
	WORKDIR /opt
	RUN git clone https://github.com/spacepy/spacepy.git
	WORKDIR spacepy
	RUN python3 setup.py install
	RUN pip3 install tensorboard future
	RUN pip3 install freetype-py pyglet
	WORKDIR /
	RUN git clone -b computer_vision https://github.com/Hamoon1987/SPINH.git
	WORKDIR /SPINH
	RUN pip3 install -r requirements.txt
	RUN ./fetch_data.sh

3- Build the docker image based on the dockerfile: docker image build -t my_spinh .  
4- Run the container from the image: docker container run -d -it --gpus all --name my_spinh spinh  
5- Start the docker container: docker start my_spinh  
6- Connect to the container  
7- Run the demo: python3 demo.py  
8- Add the dataset: 3DPW to data/3DPPW  
9- Add the new 3dpw_test.npz: python3 datasets/preprocess/pw3d.py  
10- Copy SMPL_male amd SMPL_female  
11- Run occlusion_analysis_joint.py  
	 To move the occluder on an image and get the error (179: batch_idx = 0, is the image number)  
12- Run python3 occlusion_analysis_joint_l.py --checkpoint=data/model_checkpoint.pt --dataset=3dpw --joint=13  
	To occlude a joint (--joint=13) throughot the dataset and calculate the mean error.  
13- Get the mean error for all joints from 12 and Run python3 occlusion_mesh:  
	To visualize the model sensitivity on parts  
14- Run python3 sp_op_mesh.py:  
	To project OpenPose confidence on the mesh  
15- Run python3 opConfidence.py --checkpoint=data/model_checkpoint.pt --dataset=3dpw --log_freq=20:  
	To calculate the correlation of OpenPose confidence and SPIN error throughout the 3dpw dataset  
