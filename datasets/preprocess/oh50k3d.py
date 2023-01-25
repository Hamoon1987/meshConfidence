import numpy as np
import json
import cv2

f = open("/SPINH/data/3DOH50K/annots.json")
data = json.load(f)
scaleFactor = 1.2
imgnames_ = []
scales_ = []
centers_ = []
parts_ = []
pose_ = []
shapes_ = []

for i in data.keys():
    pose = data[i]['pose']
    betas = data[i]['betas']
    bbox = data[i]["bbox"]
    bbox = [item for sublist in bbox for item in sublist]
    center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
    scale = scaleFactor*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200
    part = np.array(data[i]["smpl_joints_2d"])
    S24 = np.zeros([24,3])
    S24[:, :2] = part
    S24[:, 2] = 1
    imgnames_.append("images/" + i + ".jpg")
    scales_.append(scale)
    centers_.append(center)
    parts_.append(S24)
    pose_.append(pose[0])
    shapes_.append(betas[0])

np.savez("/SPINH/data/dataset_extras/3doh50k_test", imgname=imgnames_,
                    center=centers_,
                    scale=scales_,
                    part=parts_,
                    pose=pose_,
                    shape=shapes_)


# smpl_joints_2d = data["00000"]["smpl_joints_2d"]
# print(smpl_joints_2d)
# lsp_joints_2d = data["00000"]["lsp_joints_2d"]
# image = cv2.imread("C://Users//100844431//Master//Thesis//Dataset//3DOH50K//images//00500.jpg")
# for i in range(24):
#     cv2.circle(image, (int(smpl_joints_2d[i][0]), int(smpl_joints_2d[i][1])), 9, color = (0, 255, 0), thickness=-1)
# for i in range(14):
#     cv2.circle(image, (int(lsp_joints_2d[i][0]), int(lsp_joints_2d[i][1])), 9, color = (255, 255, 0), thickness=-1)
# cv2.imwrite('test.png', image)