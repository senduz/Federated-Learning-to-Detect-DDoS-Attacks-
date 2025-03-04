import argparse
import warnings
from collections import OrderedDict

from flwr.client import NumPyClient, ClientApp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDoSNet(nn.Module):
    def __init__(self, input_size):
        super(DDoSNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 3)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

def train(model, train_loader, device, epochs):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (data, target) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"), 1):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            if torch.isnan(loss):
                print(f"Encountered NaN in loss, skipping the update. Data: {data}, Target: {target}")
                continue
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss / i}")

def test(model, test_loader, device):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(test_loader, desc="Testing"), 1):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            if torch.isnan(loss):
                print(f"Encountered NaN in loss during testing. Data: {data}, Target: {target}")
                continue
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = correct / len(test_loader.dataset)
    print(f"Test Loss: {total_loss / i}, Accuracy: {accuracy}")
    return total_loss, accuracy

def load_data(csv_path, partition_id, total_partitions):
    df = pd.read_csv(csv_path)
    df = df.replace([float('inf'), float('-inf')], float('nan')).dropna()
    
    X = df.drop(columns=['Encoded Label'])
    y = df['Encoded Label']
    
    X_mean = X.mean()
    X_std = X.std()
    X_std[X_std == 0] = 1  
    X = (X - X_mean) / X_std
    
    X = torch.tensor(X.values, dtype=torch.float32)
    y = torch.tensor(y.values, dtype=torch.long)
    
    partition_size = len(X) // total_partitions
    start_idx = partition_id * partition_size
    end_idx = start_idx + partition_size if partition_id < total_partitions - 1 else len(X)
    
    X_partition = X[start_idx:end_idx]
    y_partition = y[start_idx:end_idx]
    
    split_idx = int(len(X_partition) * 0.8)
    X_train, y_train = X_partition[:split_idx], y_partition[:split_idx]
    X_test, y_test = X_partition[split_idx:], y_partition[split_idx:]
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)
    
    return train_loader, test_loader

parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--partition_id",
    choices=[0, 1],
    default=0,
    type=int,
    help="Partition of the dataset divided into 5 partitions."
)
args = parser.parse_args()
feature_size = 77 
net = DDoSNet(input_size=feature_size).to(DEVICE)
train_loader, test_loader = load_data("UDP_Clean.csv", partition_id=args.partition_id, total_partitions=2)

class FlowerClient(NumPyClient):
    def __init__(self, model, train_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.train_loader, DEVICE, epochs=1)
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.test_loader, DEVICE)
        print(f"Round Loss: {loss}, Accuracy: {accuracy}")
        return loss, len(self.test_loader.dataset), {"accuracy": accuracy}

def client_fn(cid: str):
    return FlowerClient().to_client()

app = ClientApp(
    client_fn=client_fn,
)

if __name__ == "__main__":
    from flwr.client import start_client

    start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(net, train_loader, test_loader).to_client(),
    )
