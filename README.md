# Confidence on Mesh
The goal of this project is to add confidence to the generated mesh by the SPIN model. A sample is shown in the figure below, where the occlusion forces the SPIN model to estimate inaccurate mesh. However, our model detects the inaccurate parts of the mesh.

<p align="center">
	<img width="400" height="200" src="teaser.png">
</p>

## To run the demo:
1. Go to your server and create a folder meshConfidence  
2. Create a docker file with the content.  
3. Create the docker image: ```docker image build -t confidence .```  
4. Create and run the container: ```docker run -it -d --gpus all --name my_confidence confidence```  
5. Attach to the running container and open meshConfidence folder  
6. Download the body_pose_model.pth from [here](https://github.com/Hzzone/pytorch-openpose) and add to openpose/models  
7. Get the smpl_vert_segmentation.json from [here](https://github.com/Meshcapade/wiki/tree/main/assets/SMPL_body_segmentation/smpl) and put it in data folder  
8. Get the pretrained MC and WJC:  
    *```gdown https://drive.google.com/uc?id=1-CIm4wxL7dmMy6BD__f83gBfgzrEq6PM  -O classifier/mesh/classifier.pt```  
    *```gdown https://drive.google.com/uc?id=1-Ndd8-dspqyHMpTTfpN05ADqPjwrNlOp -O classifier/wj/classifier_wj.pt```  
9. Now you can run the demo and choose a cropped and centered image as input. The result will be in demo folder
    *```python3 demo/demo_confidence.py --checkpoint=data/model_checkpoint.pt --img=demo/3doh_img_0_orig.png```

