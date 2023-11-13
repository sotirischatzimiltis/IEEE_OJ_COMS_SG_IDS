import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from imbens.metrics import classification_report_imbalanced
from tabulate import tabulate
import numpy as np
import torch.nn.init as init


class LeNetComplete(nn.Module):
    """CNN based on the classical LeNet architecture, but with ReLU instead of
    tanh activation functions and max pooling instead of subsampling."""
    def __init__(self):
        super(LeNetComplete, self).__init__()

        # First block - convolutional
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # Second block - convolutional
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # Third block - fully connected
        self.block3 = nn.Sequential(
            nn.Linear(in_features=256, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10))

    def forward(self, x):
        """Define forward pass of CNN

        Args:
            x: Input Tensor

        Returns:
          x: Output Tensor
        """
        # Apply first convolutional block to input tensor
        x = self.block1(x)

        # Apply second convolutional block to input tensor
        x = self.block2(x)

        # Flatten output
        x = x.view(-1, 4*4*16)

        # Apply first fully-connected block to input tensor
        x = self.block3(x)

        return F.log_softmax(x, dim=1)


class LeNetClientNetwork(nn.Module):
    """CNN following the architecture of:
    https://medium.com/ml2vec/intro-to-pytorch-with-image-classification-on-a- \
            fashion-clothes-dataset-e589682df0c5

    The ClientNetwork is used for Split Learning and implements the CNN
    until the first convolutional layer."""
    def __init__(self):
        super(LeNetClientNetwork, self).__init__()

        # First block - convolutional
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        """Defines forward pass of CNN until the split layer, which is the first
        convolutional layer

        Args:
            x: Input Tensor

        Returns:
          x: Output Tensor
        """
        # Apply first convolutional block to input tensor
        x = self.block1(x)

        return x


class LeNetServerNetwork(nn.Module):
    """CNN following the architecture of:
    https://medium.com/ml2vec/intro-to-pytorch-with-image-classification-on-a- \
            fashion-clothes-dataset-e589682df0c5

    The ServerNetwork is used for Split Learning and implements the CNN
    from the split layer until the last."""
    def __init__(self):
        super(LeNetServerNetwork, self).__init__()

        # Second block - convolutional
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # Third block - fully connected
        self.block3 = nn.Sequential(
            nn.Linear(in_features=256, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10))

    def forward(self, x):
        """Defines forward pass of CNN from the split layer until the last

        Args:
            x: Input Tensor

        Returns:
          x: Output Tensor
        """
        # Apply second convolutional block to input tensor
        x = self.block2(x)

        # Flatten output
        #x = x.view(-1, 4*4*16)
        x = x.view(x.size(0), -1)

        # Apply fully-connected block to input tensor
        x = self.block3(x)

        return x


""" FEED FORWARD NEURAL NETWORK """


class FeedForwardNNComplete(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(FeedForwardNNComplete, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        # Define the layers
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.hidden_layer1 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.hidden_layer2 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.hidden_layer3 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.hidden_layer4 = nn.Linear(hidden_sizes[3], output_size)
        self.relu = nn.ReLU()

        # Initialize weights
        init.xavier_uniform_(self.input_layer.weight)
        init.xavier_uniform_(self.hidden_layer1.weight)
        init.xavier_uniform_(self.hidden_layer2.weight)
        init.xavier_uniform_(self.hidden_layer3.weight)
        init.xavier_uniform_(self.hidden_layer4.weight)

    def forward(self, x):
        # Forward pass with ReLU activations
        x = self.relu(self.input_layer(x))
        x = self.relu(self.hidden_layer1(x))
        x = self.relu(self.hidden_layer2(x))
        x = self.relu(self.hidden_layer3(x))
        x = self.relu(self.hidden_layer4(x))
        return F.log_softmax(x, dim=1)


# Define the neural network for client
class FeedForwardNNClient(nn.Module):
    def __init__(self):
        super(FeedForwardNNClient, self).__init__()
        self.input_size = 121
        self.hidden_sizes = [140, 300, 300, 40]
        self.output_size = 5
        # Define the layers
        self.input_layer = nn.Linear(self.input_size, self.hidden_sizes[0])
        #self.hidden_layer1 = nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])
        self.relu = nn.ReLU()

        # Initialize weights
        init.xavier_uniform_(self.input_layer.weight)
        #init.xavier_uniform_(self.hidden_layer1.weight)

    def forward(self, x):
        # Forward pass with ReLU activations
        x = self.input_layer(x)
        x = self.relu(x)
        # x = self.hidden_layer1(x)
        # x = self.relu(x)
        return x


# Define the neural network for client
class FeedForwardNNServer(nn.Module):
    def __init__(self):
        super(FeedForwardNNServer, self).__init__()
        self.input_size = 121
        self.hidden_sizes = [140, 200, 200, 40]
        self.output_size = 5

        # Define the layers
        self.hidden_layer1 = nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])
        self.hidden_layer2 = nn.Linear(self.hidden_sizes[1], self.hidden_sizes[2])
        self.hidden_layer3 = nn.Linear(self.hidden_sizes[2], self.hidden_sizes[3])
        self.output_layer = nn.Linear(self.hidden_sizes[3], self.output_size)
        self.relu = nn.ReLU()

        init.xavier_uniform_(self.hidden_layer1.weight)
        init.xavier_uniform_(self.hidden_layer2.weight)
        init.xavier_uniform_(self.hidden_layer3.weight)
        init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x):
        # Forward pass with ReLU activations
        x = self.relu(self.hidden_layer1(x))
        x = self.relu(self.hidden_layer2(x))
        x = self.relu(self.hidden_layer3(x))
        x = self.output_layer(x)
        return F.log_softmax(x, dim=1)


""" FEED FORWARD NEURAL NETWORK NO LABEL SHARING"""


# Define the neural network for client
class FeedForwardNNClientNols(nn.Module):
    def __init__(self):
        super(FeedForwardNNClientNols, self).__init__()
        self.input_size = 121
        self.hidden_sizes = [140, 100, 100, 40]
        self.output_size = 5

        # Define the layers
        self.input_layer = nn.Linear(self.input_size, self.hidden_sizes[0])
        self.output_layer = nn.Linear(self.hidden_sizes[3], self.output_size)
        self.relu = nn.ReLU()

        init.xavier_uniform_(self.input_layer.weight)
        init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x):
        # Forward pass with ReLU activations
        x_1 = self.input_layer(x)
        x_2 = self.relu(x_1)
        return x_2

    def final_layer_forward(self, x):
        x_5 = self.output_layer(x)
        return F.log_softmax(x_5, dim=1)


# Define the neural network for client
class FeedForwardNNServerNols(nn.Module):
    def __init__(self):
        super(FeedForwardNNServerNols, self).__init__()
        self.input_size = 121
        self.hidden_sizes = [140, 100, 100, 40]
        self.output_size = 5

        # Define the layers
        self.hidden_layer1 = nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])
        self.hidden_layer2 = nn.Linear(self.hidden_sizes[1], self.hidden_sizes[2])
        self.hidden_layer3 = nn.Linear(self.hidden_sizes[2], self.hidden_sizes[3])
        self.relu = nn.ReLU()

        init.xavier_uniform_(self.hidden_layer1.weight)
        init.xavier_uniform_(self.hidden_layer2.weight)
        init.xavier_uniform_(self.hidden_layer3.weight)

    def forward(self, x):
        # Forward pass with ReLU activations
        x = self.relu(self.hidden_layer1(x))
        x = self.relu(self.hidden_layer2(x))
        x = self.relu(self.hidden_layer3(x))
        return x


""" Performance Evaluation Script"""


def perf_evaluation(y_true, y_pred, class_labels, name, plot=True):
    # # Compute the confusion matrix
    matrix = confusion_matrix(y_true, y_pred, normalize='true')
    conf_matrix = confusion_matrix(y_true, y_pred)
    if plot:
        print("Confusion Matrix:")
        # Plot the confusion matrix as a heatmap
        plt.figure(figsize=(15, 15))
        sns.heatmap(matrix, annot=True, fmt='.4f', cmap="viridis", square=True,
                    xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        path = str(name) + '_cf.png'
        plt.savefig(path)
        plt.close()
    # Accuracy and Classification Report
    accuracy = accuracy_score(y_true, y_pred) * 100
    print("Total Accuracy: ", accuracy)
    report = classification_report(y_true, y_pred)
    # Classification Report based on the imbens library
    report_imb = classification_report_imbalanced(y_true, y_pred)
    if plot:
        print("Classification Report")
        print(report)
        print("Classification Report based on imbens")
        print(report_imb)
        lines = report_imb.strip().split('\n')  # Split string by lines
        headers = lines[0].split()  # extracts the headers
        data_list = []
        for line in lines[2:-2]:
            values = line.split()  # Split the line by spaces
            metrics = [float(value) for value in values[1:]]  # Convert numeric values to float
            data_list.append(metrics)  # Add metrics to the data array
        data_array = np.array(data_list)
        table = tabulate(data_array, headers=headers, tablefmt='plain', floatfmt=".3f")
        output_file_path = str(name) + '.txt'
        # with open(output_file_path, "w") as file:
        #     file.write(table)
        # print("Table written to", output_file_path)
        print(table)
        # Calculate TP, TN, FP, FN for each class
        class_tp = {}
        class_tn = {}
        class_fp = {}
        class_fn = {}
        for i, label in enumerate(class_labels):
            tp = conf_matrix[i, i]
            tn = conf_matrix.sum() - conf_matrix[i, :].sum() - conf_matrix[:, i].sum() + tp
            fp = conf_matrix[:, i].sum() - tp
            fn = conf_matrix[i, :].sum() - tp
            class_tp[label] = tp
            class_tn[label] = tn
            class_fp[label] = fp
            class_fn[label] = fn
        # Print true positives, true negatives, false positives, and false negatives for each class
        for label in class_labels:
            print(f"Class {label}:")
            print("True Positives (TP):", class_tp[label])
            print("True Negatives (TN):", class_tn[label])
            print("False Positives (FP):", class_fp[label])
            print("False Negatives (FN):", class_fn[label])
            print()
    return matrix, report_imb, accuracy





