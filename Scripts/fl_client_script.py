import sys
import warnings
from collections import OrderedDict
import flwr as fl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
from models import FeedForwardNNComplete


def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc        


def train(network, trainldr, epochs):
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0005)
    criterion = torch.nn.CrossEntropyLoss()
    """Train the model on the training set."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = network.to(device)
    criterion = criterion.to(device)
    total_loss, total_acc = 0.0, 0.0
    step = 0
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_acc = 0
        train_batch = iter(trainldr)
        total = 0
        for data, targets in tqdm(train_batch):
            step += 1
            total += 1
            data = data.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            y_pred = network.forward(data)
            loss = criterion(y_pred, targets)
            acc = calculate_accuracy(y_pred, targets)
            total_loss += loss.item()
            total_acc += acc.item()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    total_loss = total_loss / (epochs * step)
    total_acc = total_acc / (epochs * step)
    return total_loss, total_acc


def test(model, loader):
    """Validate the model on the validation set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    step = 0
    with torch.no_grad():
        for data, targets in tqdm(loader):
            step += 1
            data = data.to(device)
            targets = targets.to(device)
            y_pred = model(data)
            loss += criterion(y_pred, targets).item()
            total += targets.size(0)
            correct += (torch.max(y_pred.data, 1)[1] == targets).sum().item()
    return loss / step, correct / total
    

class MyDataset(Dataset):

    def __init__(self, file_name):
        print("Collecting Data")
        df = pd.read_csv(file_name)
        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        # scaler = MinMaxScaler()
        # x = scaler.fit_transform(x)

        self.x_train = torch.tensor(x, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]
    

def load_data(train_dataset, test_dataset, bs):
    trainDs = MyDataset(train_dataset)
    testDs = MyDataset(test_dataset)
    train_loader = DataLoader(trainDs, batch_size=bs, shuffle=True, drop_last=True)
    test_loader = DataLoader(testDs, batch_size=bs, shuffle=True, drop_last=False)
    return train_loader, test_loader


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 1024
input_size = 121
hidden_size = [200, 1000, 1000, 200]
num_classes = 5
net = FeedForwardNNComplete(input_size=121, hidden_sizes=hidden_size, output_size=num_classes)

if len(sys.argv) != 3:
    print("Usage: python fl_client_script.py <trainfile> <testfile>")
    exit()
TRAIN_FILE = sys.argv[1]
VAL_FILE = sys.argv[2]

trainloader, valloader = load_data(TRAIN_FILE, VAL_FILE, bs=batch_size)
train_losses = []


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    # return the model weight as a list of NumPy ndarrays
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    # update/set the local model weights with the parameters received from the server
    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    # set the local model weights, train the local model,receive the updated local model weights
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_loss, train_acc = train(net, trainloader, epochs=1)
        train_losses.append(train_loss)
        return self.get_parameters(config={}), len(trainloader.dataset), {"train_loss": train_loss,
                                                                          "train_acc": train_acc}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, valloader)
        # print("Loss: ", loss)
        # print("Accuracy: ", accuracy)
        return loss, len(valloader.dataset), {"accuracy": accuracy}


# Start Flower client
# fl.client.start_numpy_client("localhost:8080", client=FlowerClient(),
# root_certificates=Path(".cache/certificates/ca.crt").read_bytes(),)
fl.client.start_numpy_client(server_address="localhost:8080", client=FlowerClient())
print(train_losses)
