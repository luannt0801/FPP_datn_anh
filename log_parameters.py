import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        self.n_cls = 10
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 4 * 4, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, self.n_cls)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x1 = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x1))
        protos = self.fc3(x)
        output = F.log_softmax(protos, dim=1)
        return output, protos

trans_mnist_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])

trans_mnist_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])

# Load MNIST dataset
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=trans_mnist_train)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=trans_mnist_test)

trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

def test(model, test_loader, criterion, idx):
    model.eval()
    test_loss = 0
    correct = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for data, target in tqdm.tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            all_labels.extend(target.cpu().numpy())
            all_predictions.extend(pred.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.3f}%')
    logger.info(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.3f}%')
    
    if idx == 10:  # Only plot confusion matrix for the last round (idx == 3)
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

    return accuracy, test_loss

model = Lenet().to(device)
criterion = nn.CrossEntropyLoss()

round = 10

round_dict_FPP = {}
round_dict_Avg = {}

for i in range(round):
    model.load_state_dict(torch.load(f"model_round/model_server_round_{i+1}.pt", map_location=device))
    accuracy, test_loss = test(model, testloader, criterion, i+1)
    round_dict_FPP[f"round_{i+1}"] = {"eval_loss": test_loss, "accuracy": accuracy}


for i in range(round):
    model.load_state_dict(torch.load(f"model_server_round_{i+1}.pt", map_location=device))
    accuracy, test_loss = test(model, testloader, criterion, i+1)
    round_dict_Avg[f"round_{i+1}"] = {"eval_loss": test_loss, "accuracy": accuracy}

if __name__ == "__main__":
    # Extract accuracy and avg_loss values from round_dict
    accuraciesFPP = [round_dict_FPP[f"round_{i+1}"]["accuracy"] for i in range(round)]
    avg_lossesFPP = [round_dict_FPP[f"round_{i+1}"]["eval_loss"] for i in range(round)]

    accuraciesAvg = [round_dict_Avg[f"round_{i+1}"]["accuracy"] for i in range(round)]
    avg_lossesAvg = [round_dict_Avg[f"round_{i+1}"]["eval_loss"] for i in range(round)]

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Plot accuracy
    axs[0].plot(range(1, round + 1), accuraciesFPP, label='FPP Accuracy', marker='o')
    axs[0].plot(range(1, round + 1), accuraciesAvg, label='FedAvg Accuracy', marker='x')
    axs[0].set_title('Accuracy over rounds')
    axs[0].set_xlabel('Round')
    axs[0].set_ylabel('Accuracy (%)')
    axs[0].set_xticks(range(1, round + 1))
    axs[0].grid(True)
    axs[0].legend()

    # Plot average loss
    axs[1].plot(range(1, round + 1), avg_lossesFPP, label='FPP Loss', marker='o', color='blue')
    axs[1].plot(range(1, round + 1), avg_lossesAvg, label='FedAvg Loss', marker='x', color='red')
    axs[1].set_title('Average Loss over rounds')
    axs[1].set_xlabel('Round')
    axs[1].set_ylabel('Average Loss')
    axs[1].set_xticks(range(1, round + 1))
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    #plt.savefig('metrics.png')
    plt.show()

