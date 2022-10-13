import itertools
from datasets import BaseDataset
from torch.utils.data import DataLoader
import torch
import cv2
import constants
from utils.imutils import transform
# from utils.geometry import perspective_projection
import numpy as np
from tqdm import tqdm
# # Load the dataloader
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# dataset_name = '3dpw'
# dataset = BaseDataset(None, dataset_name, is_train=False)
# batch_size = 2
# data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
# # for batch_idx, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader))):
# #     print(batch['imgname'][-1])



# batch_idx = 0
# batch = next(itertools.islice(data_loader, batch_idx, None))
# print(batch['imgname'])
# image = batch['img']
# image = image * torch.tensor([0.229, 0.224, 0.225], device=image.device).reshape(1, 3, 1, 1)
# image = image + torch.tensor([0.485, 0.456, 0.406], device=image.device).reshape(1, 3, 1, 1)
# # Preparing the image for visualization
# img = image[0].permute(1,2,0).cpu().numpy().copy()
# img = 255 * img[:,:,::-1]
# cv2.imwrite('test.jpg', img)
data = np.load("data/dataset_extras/3dpw_test_m.npz")
print(data.files)
print(data['trans'][0])