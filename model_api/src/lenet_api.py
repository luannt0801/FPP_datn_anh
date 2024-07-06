import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
import random
import pandas as pd
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import seaborn as sns 
from collections import OrderedDict
from glob_inc.utils import *
from glob_inc.add_config import *

NUM_DEVICE = server_config["NUM_DEVICE"]
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
        #x = F.log_softmax(self.fc3(x), dim=1)
        return output, protos
    
def get_dataset():
    if server_config['dataset'] == "mnist":

        trans_mnist_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

        trans_mnist_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        
        # Load MNIST dataset
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=trans_mnist_train)
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=trans_mnist_test)


        # Tách dữ liệu train theo tỉ lệ 9:1
        train_size = int(0.9 * len(train_dataset))
        prototype_size = len(train_dataset) - train_size
        train_dataset, prototype_dataset = random_split(train_dataset, [train_size, prototype_size])

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        prototype_loader = DataLoader(prototype_dataset, batch_size=64, shuffle=True)

    return train_loader, test_loader, prototype_loader, train_dataset

def sample_mnist_data(dataset, NUM_DEVICE):
    if server_config['distribution_data'] == 'iid':

        num_items = int(len(dataset)/NUM_DEVICE)
        dict_users, all_idxs = {}, [i for i in range(len(dataset))]
        for i in range(NUM_DEVICE):
            dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                                replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i])
            
    elif server_config['distribution_data'] == 'noniid':
        dict_users = {i: np.array([], dtype='int64') for i in range(NUM_DEVICE)}
        labels = np.array(dataset.dataset.targets)[dataset.indices]
        idxs = np.arange(len(labels))
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]
        # Ensure each client has at least one sample from each class
        min_samples_per_class = 1
        NUM_CLASSES = len(np.unique(labels))
        for label in range(NUM_CLASSES):
            label_idxs = idxs[idxs_labels[1] == label]
            np.random.shuffle(label_idxs)
            for i in range(NUM_DEVICE):
                selected_idxs = label_idxs[i * min_samples_per_class: (i + 1) * min_samples_per_class]
                dict_users[i] = np.concatenate((dict_users[i], selected_idxs), axis=0)
        
        # Distribute remaining samples
        remaining_idxs = idxs[NUM_DEVICE * NUM_CLASSES:]
        np.random.shuffle(remaining_idxs)

        # Calculate the number of samples for each client
        samples_per_client = len(remaining_idxs) // NUM_DEVICE
        for i in range(NUM_DEVICE):
            start_idx = i * samples_per_client
            end_idx = (i + 1) * samples_per_client if i != NUM_DEVICE - 1 else len(remaining_idxs)
            dict_users[i] = np.concatenate((dict_users[i], remaining_idxs[start_idx:end_idx]), axis=0)
        for i in range(NUM_DEVICE):
            print(f"Client {i} has {len(dict_users[i])} samples.")
    
    return dict_users

def get_dataloader_for_client(train_dataset, dict_users, client_id):
    number_client = client_id.split('_')[1]
    number_client = client_id.split('_')[1]
    data_loaders = []
    
    for client_key, indices in dict_users.items():
        if number_client == str(client_key):
            if isinstance(indices, set):
                indices = list(indices)  # Convert set to list if needed
            
            user_sampler = SubsetRandomSampler(indices)
            user_data_loader = DataLoader(train_dataset, batch_size=64, sampler=user_sampler)
            data_loaders.append(user_data_loader)
            break  # Dừng vòng lặp khi tìm thấy client_id tương ứng
    
    if not data_loaders:
        print(f"Không tìm thấy client {client_id} trong dict_users.")
    
    return data_loaders

def train_mnist(client_data_loaders, test_loader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = 0.00001
    epochs = 1

    model = Lenet().to(device)
    model.load_state_dict(torch.load("newmode.pt", map_location=device))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    print("client data loader: ",client_data_loaders)
    prototypes = {}

    for epoch in range(1, epochs + 1):
        model.train()
        for user_loader in tqdm(client_data_loaders):
            for batch_idx, (data, target) in enumerate(user_loader):
                data, target = data.to(device), target.to(device)
                output, protos = model(data)
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    for j in range(target.size(0)):
                        label = target[j].item()
                        if label not in prototypes:
                            prototypes[label] = (protos[j], 1)  
                        else:
                            prototype, count = prototypes[label]
                            prototypes[label] = (prototype + protos[j], count + 1) 
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output, _ = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Epoch: {epoch}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.3f}%')
 
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': test_loss,
        'accuracy': accuracy,
        'prototypes': prototypes
    }, "mymodel.pt")
    # Normalize prototypes 
    for label in prototypes:
        protos, count = prototypes[label]
        prototypes[label] = protos / count
    prototypes = {label: prototypes[label].tolist() for label in prototypes}
    return model.state_dict(), prototypes


'''
    Calculate Prototypes
'''
def calculate_server_prototypes(prototype_loader):
    server_prototypes = {}
    model = Lenet()
    checkpoint = torch.load('mymodel.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(prototype_loader):
            data, target = data.to(device), target.to(device)
            output, protos = model(data)
            for j in range(protos.size(0)):
                label = target[j].item()  
                if label not in server_prototypes:
                    server_prototypes[label] = (protos[j], 1)
                else:
                    prototype, count = server_prototypes[label]
                    server_prototypes[label] = (prototype + protos[j], count + 1)
    for label in server_prototypes:
        protos, count = server_prototypes[label]
        server_prototypes[label] = protos / count
    # server_prototypes = {label: server_prototypes[label].tolist() for label in server_prototypes}
    torch.save(server_prototypes, 'server_prototypes.pt')
    
    return server_prototypes

def calculate_prototype_distance(client_trainres_protos, n_round, server_prototypes):
    server_proto = {label: torch.tensor(prototype_vector) for label, prototype_vector in server_prototypes.items()}
                    #{label:(proto_server) for label, proto_server in server_prototypes.items()}
    clients_proto = {client_id: {label: torch.tensor(prototype_vector) for label, prototype_vector in prototypes.items()} for client_id, prototypes in client_trainres_protos.items()}

    # print("\n Proto tren Server: ", server_proto)
    # print("\n Proto tren Client: ", clients_proto)
    dist_state_dict = OrderedDict()
    for client_id, protos in clients_proto.items():
        client_distances = {}
        for label in server_proto.keys():
            if label in protos:
                distance = torch.nn.functional.pairwise_distance(server_proto[label], protos[label])
                client_distances[label] = distance
            else:
                client_distances[label] = None 
        dist_state_dict[client_id] = client_distances

    print(f"Distance: {dist_state_dict}")  
    return dist_state_dict

def calculate_penalty(dist_state_dict):
    penalty_lambda = {}
    for client_id, distances_dict in dist_state_dict.items():
        client_penalty = {}
        for label, distance in distances_dict.items():
            if distance is not None and distance != 0:
                client_penalty[label] = 1 / distance
            else:
                client_penalty[label] = 1  
        penalty_lambda[client_id] = client_penalty
    return penalty_lambda


'''
Running here
'''

def start_trainning_mnist():
    train_loader, test_loader, prototype_loader, train_dataset = get_dataset()
    dict_data_users = sample_mnist_data(dataset=train_dataset, NUM_DEVICE=NUM_DEVICE)
    # print("------------------------------")
    # print(dict_data_users)
    client_data_loader = get_dataloader_for_client(client_id="client_0", train_dataset= train_dataset, dict_users=dict_data_users)
    models_statedict, prototypes = train_mnist(client_data_loader)

    return models_statedict, prototypes
