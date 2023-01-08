
import sys
sys.path.insert(0, '/SPINH')
from datasets import BaseDataset
from torch.utils.data import DataLoader
import itertools
import cv2
import torch
import constants
from pytorchopenpose.src.body import Body
import numpy as np
import config
from models import hmr, SMPL
from utils.geometry import perspective_projection

def denormalize(images):
    # De-normalizing the image
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    images = images * torch.tensor([0.229, 0.224, 0.225], device=device).reshape(1, 3, 1, 1)
    images = images + torch.tensor([0.485, 0.456, 0.406], device=device).reshape(1, 3, 1, 1)
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    images = 255 * images[:, :,:,::-1]
    return images

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
dataset = BaseDataset(None, "h36m-p2", is_train=False)
batch_size = 16
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
step = 0
#702
batch = next(itertools.islice(data_loader, step, None))
image = batch['img'].to(device)
pose_3d = batch['pose_3d'][0]
pose_3d = pose_3d[pose_3d[:,3]==1].to(device)
print(batch.keys())
print(pose_3d)




image_ = denormalize(image)[0]
cv2.imwrite(f'sp_op/h36_test.png', image_)