import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm

def set_seed(seed: int):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  

set_seed(42)  # Set a fixed seed for reproducibility


learning_rate=0.01

# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.fc1 = nn.Linear(64 * 7 * 7, 128)
#         self.fc2 = nn.Linear(128, 10)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, 2)
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2)
#         x = torch.flatten(x, 1)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
    
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.dropout = nn.Dropout(0.4)
        
        self.fc1 = nn.Linear(256*7*7, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def evaluate_model(model, data_loader,losses_per_set,index,data_loader_train):
    model.eval()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in data_loader:
            data=data.to("cuda")
            target=target.to("cuda")
            output = model(data)
            
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            # loss = criterion(output, target)
            # losses_per_set[index].append(loss.item())
    with torch.no_grad():
        for data, target in data_loader_train:
            data=data.to("cuda")
            target=target.to("cuda")
            output = model(data)
            loss = criterion(output, target)
            losses_per_set[index].append(loss.item())
    return 100 * correct / total

def client_compute_gradients(model, data_loader):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    gradients = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    for data, target in data_loader:
        optimizer.zero_grad()
        data=data.to("cuda")
        target=target.to("cuda")
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        # Collect gradients
        for name, param in model.named_parameters():
            gradients[name] = param.grad.detach().clone()
    return gradients

transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset= datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# Split into subsets
test_subsets = []
train_subsets = []
for i in tqdm(range(0, 10, 2)):
    indices = [idx for idx, (img, label) in enumerate(trainset) if label in [i, i+1]]
    train_subsets.append(torch.utils.data.Subset(trainset, indices))
    indices2 = [idx for idx, (img, label) in enumerate(testset) if label in [i, i+1]]
    test_subsets.append(torch.utils.data.Subset(testset, indices2))
train_loaders = [DataLoader(train_subset, batch_size=len(train_subset), shuffle=True) for train_subset in train_subsets]
test_loaders = [DataLoader(test_subset, batch_size=len(test_subset), shuffle=True) for test_subset in test_subsets]


losses_per_set=[]
for i in range(5):
    losses_per_set.append([])

global_model=SimpleCNN()
global_model.to("cuda")

server_optimizer = optim.SGD(global_model.parameters(), lr=learning_rate)
client_gradients=[client_compute_gradients(global_model, loader) for loader in train_loaders]
update={name: torch.zeros_like(param) for name, param in global_model.named_parameters()}
for name,param in global_model.named_parameters():
    for index_client in range(5):
        update[name]+=client_gradients[index_client][name]
server_optimizer.zero_grad()
for name,param in global_model.named_parameters():
    param.grad=update[name]
server_optimizer.step()
for i in range(5):
    print(evaluate_model(global_model,test_loaders[i],losses_per_set,i,train_loaders[i]))
print("*********")


for k in range(15):

    client_gradients=[client_compute_gradients(global_model, loader) for loader in train_loaders]
    
    update={name: torch.zeros_like(param) for name, param in global_model.named_parameters()}
    for name,param in global_model.named_parameters():
        for index_client in range(5):
            update[name]+=client_gradients[index_client][name]

    server_optimizer.zero_grad()
    for name,param in global_model.named_parameters():
        param.grad=update[name]
    server_optimizer.step()

    for i in range(5):
        print(evaluate_model(global_model,test_loaders[i],losses_per_set,i,train_loaders[i]))
    print("*********")

for i in range(5):
    plt.plot(losses_per_set[i],label=f"Client ${i+1}$")

plt.xticks([i for i in range(16)],fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Epoch",fontsize=14)
plt.ylabel("Loss on each client's dataset",fontsize=14)
plt.legend()
plt.grid()
plt.title("No malicious client",fontsize=14)
plt.savefig("simple.pdf",
              format="pdf",
              dpi=300,
              bbox_inches="tight",
              transparent=True)
