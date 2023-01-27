import sys
sys.path.insert(0, '/SPINH')
from classifier_wj_model import Model
from classifier_wj_dataloader import Classifier_Dataset
import torch
from classifier.classifier_config import args
import matplotlib.pyplot as plt

net = Model(14, 28, 56, 28, 14)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
net = net.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr = 0.0001)
train_classifier_dataset = Classifier_Dataset("classifier/data/train/all_train.csv")
train_dataloader = torch.utils.data.DataLoader(train_classifier_dataset, batch_size = args.batch_size, shuffle = True)
val_classifier_dataset = Classifier_Dataset("classifier/data/val/all_val.csv")
val_dataloader = torch.utils.data.DataLoader(val_classifier_dataset, batch_size = args.batch_size, shuffle = True)
epochs = args.epochs
running_loss_history = []
running_acc_history = []
val_running_loss_history = []
val_running_acc_history = []
for epoch in range(epochs):
    running_loss = 0.0
    running_corrects = 0.0
    val_running_loss = 0.0
    val_running_corrects = 0.0
    for training_inputs, training_labels in train_dataloader:
        training_inputs = training_inputs.float().to(device)
        training_labels = training_labels.type(torch.LongTensor).to(device)
        # Forward 
        training_outputs = net(training_inputs)
        # Loss
        training_loss = criterion(training_outputs, training_labels)
        # Clear the gradient
        optimizer.zero_grad()
        # Backpropagation 
        training_loss.backward()
        # Update the weights
        optimizer.step()
        running_loss += training_loss.item()
        _, training_preds = torch.max(training_outputs, 1)
        running_corrects += torch.sum(training_preds == training_labels)
        
    else:
        with torch.no_grad():
            for val_inputs, val_labels in val_dataloader:
                val_inputs = val_inputs.float().to(device)
                val_labels = val_labels.type(torch.LongTensor).to(device)
                val_outputs = net(val_inputs)
                val_loss = criterion(val_outputs, val_labels)
                val_running_loss += val_loss.item()
                _, val_preds = torch.max(val_outputs, 1)
                val_running_corrects += torch.sum(val_preds == val_labels)

    # Print Epoch Statistics
    running_corrects = running_corrects.cpu().numpy()
    val_running_corrects = val_running_corrects.cpu().numpy()
    training_accuracy = (running_corrects / (len(train_dataloader)*args.batch_size))
    running_loss = (running_loss / len(train_dataloader))
    val_accuracy = (val_running_corrects / (len(val_dataloader)*args.batch_size))
    val_running_loss = (val_running_loss / len(val_dataloader))
    print('Epoch {}/{}, Training_Loss = {:.3f}, Training_Accuracy = {:.3f}'.format(epoch+1, epochs, running_loss, training_accuracy))
    print('Epoch {}/{}, Val_Loss = {:.3f}, Val_Accuracy = {:.3f}'.format(epoch+1, epochs, val_running_loss, val_accuracy))
    print()

    running_loss_history.append(running_loss)
    running_acc_history.append(training_accuracy)
    val_running_loss_history.append(val_running_loss)
    val_running_acc_history.append(val_accuracy)
    
    train_classifier_dataset = Classifier_Dataset("classifier/data/train/all_train.csv")
    val_classifier_dataset = Classifier_Dataset("classifier/data/val/all_val.csv")
    train_dataloader = torch.utils.data.DataLoader(train_classifier_dataset, batch_size = args.batch_size, shuffle = True)
    val_dataloader = torch.utils.data.DataLoader(val_classifier_dataset, batch_size = args.batch_size, shuffle = True)
# Save the model
torch.save(net.state_dict(), 'classifier/wj/classifier_wj.pt')



plt.figure(2, figsize=(8,6))
plt.plot(running_loss_history, label='training loss')
plt.plot(val_running_loss_history, label='validation loss')
plt.legend()
plt.savefig('classifier/wj/Loss_history')

plt.figure(3, figsize=(8,6))
plt.plot(running_acc_history, label='training accuracy')
plt.plot(val_running_acc_history, label='validation accuracy')
plt.legend()
plt.savefig('classifier/wj/Accuracy_history')