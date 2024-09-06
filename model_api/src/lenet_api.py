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
        test_size = int(0.3 * len(test_dataset))
        prototype_size = len(test_dataset) - test_size
        test_dataset, prototype_dataset = random_split(test_dataset, [test_size, prototype_size])

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        prototype_loader = DataLoader(prototype_dataset, batch_size=64, shuffle=True)

    return train_loader, test_loader, prototype_loader, train_dataset

# def sample_mnist_data(dataset, NUM_DEVICE):
#     dataset_users = {i: np.array([], dtype='int64') for i in range(NUM_DEVICE)}
#     targets = np.array(dataset.targets)

#     if server_config['distribution_data'] == 'iid':
#         num_samples = len(dataset) // NUM_DEVICE
#         all_indices = np.arange(len(dataset))
#         np.random.shuffle(all_indices)
#         for i in range(NUM_DEVICE):
#             dataset_users[i] = all_indices[i * num_samples: (i + 1) * num_samples]
    
#     elif server_config['distribution_data'] == 'noniid':
#         min_size = 0
#         while min_size < 10:
#             idx_batch = [[] for _ in range(NUM_DEVICE)]
#             for k in range(10):
#                 idx_k = np.where(targets == k)[0]
#                 np.random.shuffle(idx_k)
#                 proportions = np.random.dirichlet(np.repeat(0.5, NUM_DEVICE))
#                 proportions = np.array([p * (len(idx_j) < len(dataset) / NUM_DEVICE) for p, idx_j in zip(proportions, idx_batch)])
#                 proportions = proportions / proportions.sum()
#                 proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
#                 idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
#             min_size = min([len(idx_j) for idx_j in idx_batch])

#         #for client_id in range(NUM_DEVICE):
#         #    for k in range(num_class):
#         #       if not any(targets[idx] == k for idx in idx_batch[client_id]):
#         #           idx_k = np.where(targets == k) [0]
#         #            np.random.shuffle(idx_k)
#         #            idx_batch[client_id].append(idx_k[0])

#         for j in range(NUM_DEVICE):
#             np.random.shuffle(idx_batch[j])
#             dataset_users[j] = np.hstack(idx_batch[j])

#     return dataset_users

def sample_mnist_data(dataset, num_devices):
    alpha=1
    dataset_users = {i: np.array([], dtype='int64') for i in range(num_devices)}
    targets = np.array(dataset.targets)
    num_classes = 10  # MNIST có 10 nhãn (0-9)
    
    if server_config['distribution_data'] == 'iid':
        num_samples = len(dataset) // num_devices
        all_indices = np.arange(len(dataset))
        np.random.shuffle(all_indices)
        for i in range(num_devices):
            dataset_users[i] = all_indices[i * num_samples: (i + 1) * num_samples]

    elif server_config['distribution_data'] == 'noniid':
        # Chia dữ liệu non-IID với phân phối Dirichlet
        idx_batch = [[] for _ in range(num_devices)]
        
        for k in range(num_classes):
            idx_k = np.where(targets == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet([alpha] * num_devices)
            proportions = np.array([p * len(idx_k) for p in proportions])
            proportions = proportions.astype(int)
            
            # Adjust proportions to ensure sum equals to the total number of samples
            while proportions.sum() < len(idx_k):
                proportions[np.random.randint(0, num_devices)] += 1
            while proportions.sum() > len(idx_k):
                proportions[np.random.randint(0, num_devices)] -= 1
            
            split_idx_k = np.split(idx_k, np.cumsum(proportions)[:-1])
            for i in range(num_devices):
                idx_batch[i].extend(split_idx_k[i])
        
        for j in range(num_devices):
            np.random.shuffle(idx_batch[j])
            dataset_users[j] = np.array(idx_batch[j])
        
        for j in range(num_devices):
            client_labels = targets[dataset_users[j]]
            unique_labels = np.unique(client_labels)
            missing_labels = set(range(num_classes)) - set(unique_labels)
            if missing_labels:
                for label in missing_labels:
                    idx_k = np.where(targets == label)[0]
                    chosen_idx = np.random.choice(idx_k, 1)
                    dataset_users[j] = np.append(dataset_users[j], chosen_idx)

    return dataset_users

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

model = Lenet().to(device)
torch.save(model.state_dict(), "saved_model/LENETModel.pt")

def train_mnist(client_data_loaders, test_loader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = float(server_config['learning_rate'])
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
 
    # torch.save({
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'loss': test_loss,
    #     'accuracy': accuracy,
    #     'prototypes': prototypes
    # }, "mymodel.pt")

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
    # checkpoint = torch.load('/saved_model/LENETModel.pt')
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model.to(device)
    model.load_state_dict(torch.load("saved_model/LENETModel.pt", map_location=device))
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
    server_proto_tensor = torch.stack(list(server_proto.values()))
    clients_proto = {client_id: {label: torch.tensor(prototype_vector) for label, prototype_vector in prototypes.items()} for client_id, prototypes in client_trainres_protos.items()}
    clients_proto_tensor = {client_id: torch.stack(list(protos.values())) for client_id, protos in clients_proto.items()}

    print("Server Proto Tensor:", server_proto_tensor)
    for client_id, tensor in clients_proto_tensor.items():
        print(f"Client {client_id} Proto Tensor:", tensor)
    # print("\n Proto tren Server: ", server_proto)
    # print("\n Proto tren Client: ", clients_proto)
    dist_state_dict = OrderedDict()
    for client_id, protos in clients_proto_tensor.items():
        distance = torch.nn.functional.pairwise_distance(server_proto_tensor, protos)
        distance = torch.mean(distance)
        dist_state_dict[client_id] = distance
        #client_distances[client_id] = distance
        """for label in server_proto.keys():
            distance = torch.nn.functional.pairwise_distance(server_proto[label], protos[str(label)])
            distance = torch.mean(distance)
            client_distances[label] = distance
        """
        dist_state_dict[client_id] = distance.item()

    print(f"Distance: {dist_state_dict}")  
    return dist_state_dict

def calculate_penalty(dist_state_dict):
    penalty_lambda = {}
    for client_id, distance in dist_state_dict.items():
        if distance is not None and distance != 0:
            penalty_lambda[client_id] = 1 / distance
        else:
            penalty_lambda[client_id] = 1  # Hoặc có thể bỏ qua client này nếu khoảng cách không hợp lệ
    for client_id, penalties in penalty_lambda.items():
        print(f"Client {client_id} Penalty: {penalties}")

    return penalty_lambda


'''
Running here
'''

def start_trainning_mnist():
    train_loader, test_loader, prototype_loader, train_dataset = get_dataset()
    dict_data_users = sample_mnist_data(dataset=train_dataset, NUM_DEVICE=NUM_DEVICE)

    # print("------------------------------")
    print(dict_data_users)
    client_data_loader = get_dataloader_for_client(client_id="client_0", train_dataset= train_dataset, dict_users=dict_data_users)
    models_statedict, prototypes = train_mnist(client_data_loader)

    return models_statedict, prototypes
