from sklearn.utils import shuffle
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from classifier_config import args
import math


# Define the DataLoader Class

class Classifier_Dataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self._load_dataset()
    
    def _load_dataset(self):
        # Read Data
        data = pd.read_csv(self.data_path)
        self.sp_op = data.iloc[:,:14].values
        self.label_wj = data.iloc[:,15].values
        # Standardize
        sc = StandardScaler()
        self.sp_op = sc.fit_transform(self.sp_op)
        # Make tensors with equal dimensions
        self.sp_op = torch.tensor(self.sp_op, dtype=torch.float)
        self.label_wj = torch.tensor(self.label_wj, dtype=torch.float)


    def __getitem__(self, indesp_op):
        return self.sp_op[indesp_op], self.label_wj[indesp_op]

    def __len__(self):
        return len(self.sp_op)


# Gets the path to data and returns the dataloader with batch 32 
def fetch_dataloader(data_path, mode):
    classifier_dataset = Classifier_Dataset(data_path)
    if mode == "test":
        test_dataloader = torch.utils.data.DataLoader(classifier_dataset, batch_size = args.batch_size, shuffle = False)
        for (sp_op, label_wj) in test_dataloader:
            print('======================================')
            print('Train Dataloader:')
            print(f'  Batch size: {sp_op.shape[0]}')
            print(f'  Test batches: {len(test_dataloader)}')
            print('======================================')
            break
        return  test_dataloader
    else:
        train_size =  math.floor(args.split_ratio*len(classifier_dataset)) 
        val_size = len(classifier_dataset) - train_size
        train_set, val_set = torch.utils.data.random_split(classifier_dataset, [train_size, val_size])
        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size = args.batch_size, shuffle = True)
        val_dataloader = torch.utils.data.DataLoader(val_set, batch_size = args.batch_size, shuffle = True)
        # for (sp_op, label_wj) in train_dataloader:
        #     print('======================================')
        #     print('Train Dataloader:')
        #     print(f'  Batch size: {sp_op.shape[0]}')
        #     print(f'  Training batches: {len(train_dataloader)}')
        #     print(f'  label_wj shape: {label_wj.shape}')
        #     print('======================================')
        #     break
        # for (sp_op, label_wj) in val_dataloader:
        #     print('======================================')
        #     print('Dataloader:')
        #     print(f'  Batch size: {sp_op.shape[0]}')
        #     print(f'  Training batches: {len(val_dataloader)}')
        #     print(f'  label_wj shape: {label_wj.shape}')
        #     print('======================================')
        #     break
        return  train_dataloader, val_dataloader