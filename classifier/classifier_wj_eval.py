import torch
from classifier_config import args
from classifier_wj_dataloader import fetch_dataloader


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torch.load('classifier/classifier_wj.pt')
model.eval()
test_dataloader = fetch_dataloader("classifier/occ_h36m-p1_test.csv", "test")
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