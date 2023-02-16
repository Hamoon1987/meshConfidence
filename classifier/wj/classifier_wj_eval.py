import sys
sys.path.insert(0, '/SPINH')
import torch
from classifier.classifier_config import args
from classifier_wj_dataloader import Classifier_Dataset
from classifier_wj_model import classifier_wj_model


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = classifier_wj_model().to(device)
model.eval()
test_classifier_dataset = Classifier_Dataset("classifier/data/test/occ_3doh_test.csv")
test_dataloader = torch.utils.data.DataLoader(test_classifier_dataset, batch_size = args.batch_size, shuffle = False)
running_corrects=0
running_corrects_topk=0
for test_inputs, test_labels in test_dataloader:
    test_inputs = test_inputs.float().to(device)
    test_labels = test_labels.type(torch.LongTensor).to(device)
    test_outputs = model(test_inputs)
    _, test_preds = torch.max(test_outputs, 1)
    _, test_pred_topk = torch.topk(test_outputs, 3)
    running_corrects += torch.sum(test_preds == test_labels)
    correct = 0
    for i in range(len(test_inputs)):
        correct += (test_labels[i] in test_pred_topk[i])
    running_corrects_topk += correct
running_corrects = running_corrects.cpu().numpy()
test_accuracy = (running_corrects / (len(test_dataloader)*args.batch_size))
print('Test_Accuracy = {:.3f}'.format(test_accuracy))
test_accuracy_topk = (running_corrects_topk / (len(test_dataloader)*args.batch_size))
print('Test_Accuracy_Topk = {:.3f}'.format(test_accuracy_topk))