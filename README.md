# SPINH
1- Go to server creat a folder SPINH  
2- Copy the dockerfile    
3- Build the docker image based on the dockerfile: docker image build -t my_spinh .  
4- Run the container from the image: docker container run -d -it --gpus all --name my_spinh spinh  
5- Start the docker container: docker start my_spinh  
6- Connect to the container  
7- Run the demo: python3 demo.py  
8- Add the dataset: 3DPW to data/3DPPW  
9- Add the new 3dpw_test_m.npz: python3 datasets/preprocess/pw3d.py  
10- Copy SMPL_male amd SMPL_female  

To move a occluder over an image and project the MPJPE heatmap (SPIN) on it:  
	python3 occlusion_analysis.py --checkpoint=data/model_checkpoint.pt --dataset=3dpw --img_number=0  
  
To occlude a joint throughout the dataset at a time and calculate the MPJPE (SPIN) in each case:  
	python3 occlusion_analysis_joint.py --checkpoint=data/model_checkpoint.pt --dataset=3dpw  
  
To project the calculated MPJPE on the mesh:  
	python3 occlusion_mesh.py --error_list=[List of MPJPE (1x14)]  
  
To project OpenPose confidence on SPIN mesh:    
	python3 sp_op_mesh.py  
  
To calculate the correlation of OpenPose and SPIN estimation difference and SPIN error for a particular joint throughout the 3dpw dataset:  
	python3 op_sp.py --checkpoint=data/model_checkpoint.pt --dataset=3dpw --log_freq=20  