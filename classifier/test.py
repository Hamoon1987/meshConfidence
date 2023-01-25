import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
classifier = torch.load('classifier/classifier.pt')
classifier.eval()
classifier_wj = torch.load('classifier/classifier_wj.pt')
classifier_wj.eval()
input = torch.tensor([0.1137, 0.0766, 0.1002, 0.0898, 0.0640, 0.0887, 0.0814, 0.0797, 0.1089,
        0.1136, 0.0709, 0.0417, 0.0682, 0.1119], dtype=torch.float).to(device)

with torch.no_grad():
    output = classifier(input)
    output = output[0].cpu().numpy()
    print(output)
    if output > 0.5:
        output_wj = classifier_wj(input)
        print(output_wj.cpu().numpy())