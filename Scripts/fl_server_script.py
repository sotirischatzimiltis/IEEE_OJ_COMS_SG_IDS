from typing import List, Tuple
from collections import OrderedDict
import warnings
import pandas as pd
import flwr as fl
from flwr.common import Metrics
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from models import FeedForwardNNComplete, perf_evaluation
from sklearn.metrics import accuracy_score
from flwr.common import NDArrays, Scalar
from typing import Dict, Optional, Tuple
import numpy as np
import torch.nn as nn


# Dataset method
class MyDataset(Dataset):

    def __init__(self, file_name):
        df = pd.read_csv(file_name)
        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        self.x_train = torch.tensor(x, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""
    # Load data and model here to avoid the overhead of doing it in `evaluate` itself

    # The `evaluate` function will be called after every round
    def evaluate(server_round: int, parameters: NDArrays, config: Dict[str, Scalar])\
            -> Optional[Tuple[float, Dict[str, Scalar]]]:

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        bs =1024
        VAL_FILE = "validation_dataset.csv"
        TEST_FILE = "test_dataset.csv"
        valDs = MyDataset(VAL_FILE)
        testDs = MyDataset(TEST_FILE)

        val_loader = DataLoader(valDs, batch_size=bs, shuffle=True, drop_last=False)
        test_loader = DataLoader(testDs, batch_size=bs, shuffle=True, drop_last=False)

        criterion = nn.CrossEntropyLoss()

        set_parameters(model, parameters)  # Update model with the latest parameters
        val_loss, test_loss = 0.0, 0.0
        val_step, test_step = 0, 0
        val_correct, val_total = 0, 0
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for data, targets in tqdm(val_loader):
                val_step += 1
                data = data.to(device)
                targets = targets.to(device)
                y_pred = model(data)
                val_loss += criterion(y_pred, targets).item()
                val_total += targets.size(0)
                val_correct += (torch.max(y_pred.data, 1)[1] == targets).sum().item()

        val_loss = val_loss/val_step
        val_acc = val_correct/val_total

        with torch.no_grad():
            for data, targets in tqdm(test_loader):
                test_step += 1
                data = data.to(device)
                targets = targets.to(device)
                y_pred = model(data)
                test_loss += criterion(y_pred, targets).item()
                test_total += targets.size(0)
                test_correct += (torch.max(y_pred.data, 1)[1] == targets).sum().item()

        test_loss = test_loss / test_step
        test_acc = test_correct / test_total

        return val_loss, {"val_loss": val_loss, "val_acc": val_acc,
                          "test_loss": test_loss, "test_acc": test_acc}

    return evaluate


# 4. Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy from weighted_average": sum(accuracies) / sum(examples)}


# Data manipulation before FL
TRAIN_FILE = "../Datasets/smotetomektrain.csv"
TEST_FILE = "../Datasets/smotetomektest.csv"
batch_size = 1024

train_df = pd.read_csv(TRAIN_FILE)  # read train csv
test_df = pd.read_csv(TEST_FILE)  # read test csv
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)  # split train and val

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

y_train = train_df['label']
x_train = train_df.drop('label', axis=1)

y_val = val_df['label']
x_val = val_df.drop('label', axis=1)

y_test = test_df['label']
x_test = test_df.drop('label', axis=1)


# Scale values
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

val_tensor = torch.tensor(x_val, dtype=torch.float32)  # Use appropriate data type
test_tensor = torch.tensor(x_test, dtype=torch.float32)  # Use appropriate data type


# Split data to N clients and save them
clients = 7
kf = KFold(n_splits=clients, shuffle=True, random_state=42)
fold_number = 1
for train_index, val_index in kf.split(x_train):
    # Split the data into train and test sets for this fold
    x_t = x_train[val_index]
    y_t = y_train.iloc[val_index].reset_index(drop=True)
    x_df = pd.DataFrame(x_t)
    train_fold = pd.concat([x_df, y_t], axis=1).reset_index(drop=True)
    fold_csv_filename = f'client_{fold_number}_train_data.csv'
    train_fold.to_csv(fold_csv_filename, index=False)
    print(f"Saved Fold {fold_number} to {fold_csv_filename}")

    # Increment the fold number
    fold_number += 1

# Save validation dataset
val_df = pd.DataFrame(x_val)
val_set = pd.concat([val_df, y_val], axis=1).reset_index(drop=True)
fold_csv_filename = 'validation_dataset.csv'
val_set.to_csv(fold_csv_filename, index=False)
print(f"Saved validation dataset to {fold_csv_filename}")

# Save test dataset
test_df = pd.DataFrame(x_test)
test_set = pd.concat([test_df, y_test], axis=1).reset_index(drop=True)
fold_csv_filename = 'test_dataset.csv'
test_set.to_csv(fold_csv_filename, index=False)
print(f"Saved test dataset to {fold_csv_filename}")


# FL procedure
warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 1024
input_size = 121
hidden_size = [200, 1000, 1000, 200]
num_classes = 5
model = FeedForwardNNComplete(input_size=121, hidden_sizes=hidden_size, output_size=num_classes)

strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average,
                                     evaluate_fn=get_evaluate_fn(model),
                                     fraction_fit=1, min_fit_clients=7, min_available_clients=7,
                                     min_evaluate_clients=7)

# Start Flower server
# fl.server.start_server(
#     server_address="localhost:8080",
#     config={"num_rounds": 5},
#     strategy=strategy,
#     certificates=(
#         Path(".cache/certificates/ca.crt").read_bytes(),
#         Path(".cache/certificates/server.pem").read_bytes(),
#         Path(".cache/certificates/server.key").read_bytes(),
#     )
# )

fl.server.start_server(server_address="localhost:8080", config=fl.server.ServerConfig(num_rounds=20),
                       strategy=strategy)


#  Evaluation of final model on val and test set
class_labels = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']

# Val Set
correct = 0
total = 0
outputs = model(val_tensor)
_, predicted = torch.max(outputs.data, 1)
y_pred = predicted.numpy()
matrix, report_imb, accuracy = perf_evaluation(y_val, y_pred, class_labels, True)

# Test Set
print("============================= Test Set ==========================================")
correct = 0
total = 0
outputs = model(test_tensor)
_, predicted = torch.max(outputs.data, 1)
y_pred = predicted.numpy()
matrix, report_imb, accuracy = perf_evaluation(y_test, y_pred, class_labels, True)
