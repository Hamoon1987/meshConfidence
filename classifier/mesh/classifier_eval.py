import sys
sys.path.insert(0, '/SPINH')
import torch
from classifier.classifier_config import args
from classifier_dataloader import Classifier_Dataset
import numpy as np
import pandas as pd
from classifier_model import classifier_model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = classifier_model().to(device)
model.eval()
test_classifier_dataset = Classifier_Dataset("classifier/data/test/h36m_p1_test.csv")
test_dataloader = torch.utils.data.DataLoader(test_classifier_dataset, batch_size = args.batch_size, shuffle = False)
# prediction_s1 = []
running_corrects=0
for test_inputs, test_labels in test_dataloader:
    test_inputs = test_inputs.float().to(device)
    test_labels = test_labels.float().to(device)
    test_outputs = model(test_inputs)
    test_preds = (test_outputs>0.5).float()
    # prediction_s1.append(test_preds)
    running_corrects += torch.sum(test_preds == test_labels)
running_corrects = running_corrects.cpu().numpy()
test_accuracy = (running_corrects / (len(test_dataloader)*args.batch_size))
print('Test_Accuracy = {:.3f}'.format(test_accuracy))

# prediction_s1 = torch.tensor([14 * item for sublist in prediction_s1 for item in sublist])

# from classifier_wj_dataloader import fetch_dataloader
# model = torch.load('classifier/classifier_wj.pt')
# model.eval()
# test_dataloader = fetch_dataloader(args.h36m_p1_test_path, "test")
# running_corrects=0
# prediction_s2 = []
# for test_inputs, test_labels in test_dataloader:
#     test_inputs = test_inputs.float().to(device)
#     test_labels = test_labels.type(torch.LongTensor).to(device)
#     test_outputs = model(test_inputs)
#     _, test_preds = torch.max(test_outputs, 1)
#     prediction_s2.append(test_preds)
#     running_corrects += torch.sum(test_preds == test_labels)
# running_corrects = running_corrects.cpu().numpy()
# test_accuracy = (running_corrects / (len(test_dataloader)*args.batch_size))
# print('Test_Accuracy = {:.3f}'.format(test_accuracy))
# prediction_s2 = torch.tensor([item for sublist in prediction_s2 for item in sublist])

# prediction = prediction_s1.clone()
# for i in range(len(prediction_s1)):
#     if prediction[i] == 0:
#         prediction[i] = prediction_s2[i]
# df = pd.read_csv("/SPINH/classifier/h36m-p1_test.csv")
# labels = torch.tensor(df['label_c'], dtype=torch.float)
# corrects = torch.sum(prediction==labels).cpu().numpy()


# # print(prediction_s1[100:110])
# # corrects = torch.sum(prediction_s1 == labels_s1).cpu().numpy()
# print(corrects/(len(test_dataloader)*args.batch_size))