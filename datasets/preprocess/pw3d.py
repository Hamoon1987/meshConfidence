import os
import cv2
import numpy as np
import pickle

def pw3d_extract(dataset_path, out_path):

    # scale factor
    scaleFactor = 1.2

    # structs we use
    imgnames_, scales_, centers_ = [], [], []
    poses_, shapes_, genders_, translation_ = [], [], [], []
    parts_, joint_position_, trans_ = [], [], []
    camera_extrinsics_, cam_intrinsics_, bbox_ = [], [], []
    # get a list of .pkl files in the directory
    dataset_path = os.path.join(dataset_path, 'sequenceFiles', 'test')
    files = [os.path.join(dataset_path, f) 
        for f in os.listdir(dataset_path) if f.endswith('.pkl')]
    # go through all the .pkl files
    for filename in files:
        with open(filename, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            smpl_pose = data['poses']
            smpl_betas = data['betas']
            poses2d = data['poses2d']
            global_poses = data['cam_poses']
            genders = data['genders']
            valid = np.array(data['campose_valid']).astype(np.bool)
            num_people = len(smpl_pose)
            num_frames = len(smpl_pose[0])
            seq_name = str(data['sequence'])
            cam_intrinsics = data['cam_intrinsics']
            joint_position = data['jointPositions']
            trans = data['trans']
            img_names = np.array(['imageFiles/' + seq_name + '/image_%s.jpg' % str(i).zfill(5) for i in range(num_frames)])
            # get through all the people in the sequence
            for i in range(num_people):
                valid_pose = smpl_pose[i][valid[i]]
                valid_betas = np.tile(smpl_betas[i][:10].reshape(1,-1), (num_frames, 1))
                valid_betas = valid_betas[valid[i]]
                valid_keypoints_2d = poses2d[i][valid[i]]
                valid_img_names = img_names[valid[i]]
                valid_global_poses = global_poses[valid[i]]
                valid_joint_position = joint_position[i][valid[i]]
                # print(valid_cam_intrinsics.shape)
                gender = genders[i]
                # consider only valid frames
                valid_trans = trans[i]
                for valid_i in range(valid_pose.shape[0]):
                    part = valid_keypoints_2d[valid_i,:,:].T
                    part = part[part[:,2]>0,:]
                    bbox = [min(part[:,0]), min(part[:,1]),
                        max(part[:,0]), max(part[:,1])]
                    center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
                    scale = scaleFactor*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200
                    
                    # transform global pose
                    pose = valid_pose[valid_i]
                    joint_positions = valid_joint_position[valid_i]
                    # translation = valid_global_poses[valid_i][:3,3]
                    camera_extrinsics = valid_global_poses[valid_i][:3,:]
                    extrinsics = valid_global_poses[valid_i][:3,:3]
                    pose[:3] = cv2.Rodrigues(np.dot(extrinsics, cv2.Rodrigues(pose[:3])[0]))[0].T[0]
                    transs = valid_trans[valid_i]
                    

                    imgnames_.append(valid_img_names[valid_i])
                    centers_.append(center)
                    scales_.append(scale)
                    poses_.append(pose)
                    shapes_.append(valid_betas[valid_i])
                    genders_.append(gender)
                    parts_.append(part)
                    camera_extrinsics_.append(camera_extrinsics)
                    cam_intrinsics_.append(cam_intrinsics)
                    joint_position_.append(joint_positions)
                    bbox_.append(bbox)
                    trans_.append(transs)

                    # translation_.append(translation)

    # store data
    # translation_ = np.array(translation_, dtype=object)
    # print(cam_intrinsics_[0].shape)
    # print(camera_extrinsics_[0].shape)
    # print(joint_position_[0])
    # print(len(trans_))
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path,
        '3dpw_test_m.npz')
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       pose=poses_,
                       shape=shapes_,
                       gender=genders_,
                       camera_extrinsics= camera_extrinsics_,
                       camera_intrinsics = cam_intrinsics_,
                       joint_position = joint_position_,
                       bbox = bbox_,
                       trans = trans_,
                    #    global_pose = global_pose_,
                    #    translation=translation_,
                       )

pw3d_extract('./data/3DPW', './data/dataset_extras')