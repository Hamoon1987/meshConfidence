import torch
import torch.nn as nn
# from classifier_model import classifier_model
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# classifier = classifier_model().to(device)
# # torch.save(classifier.state_dict(), 'classifier/classifier.pt')
# # classifier.eval()
# # classifier_wj = torch.load('classifier/classifier_wj.pt')
# # torch.save(classifier_wj.state_dict(), 'classifier/classifier_wj.pt')
# # classifier_wj.eval()
# input = torch.tensor([50, 50, 50, 50, 0.1043, 0.1062, 0.0570, 0.0713, 0.0891,
#         0.1235, 0.0640, 0.0615, 0.0759, 0.1340], dtype=torch.float).to(device)

# with torch.no_grad():
#     output = classifier(input)
#     output = output[0].cpu().numpy()
#     print(output)
# #     if output > 0.5:
# #         output_wj = classifier_wj(input)
# #         print(output_wj.cpu().numpy())
m = nn.Softmax(dim=1)
input = torch.randn(2, 3)
output = m(input)
print(output)