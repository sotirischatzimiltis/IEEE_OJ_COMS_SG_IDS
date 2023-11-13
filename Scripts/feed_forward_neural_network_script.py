import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from time import time
import numpy as np
from sys import argv
from argparse import ArgumentParser, Namespace
from models import FeedForwardNNComplete, perf_evaluation
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
import time


"""Hyperparameters"""
input_size = 121
# hidden_size = [200, 1000, 1000, 200]
hidden_size = [140, 100, 100, 40]
num_classes = 5
learning_rate = 0.0075
batch_size = 256
num_epochs = 40
log_steps = 32

"""Data Manipulation"""
# TRAIN_FILE = "../NewDatasets/multiclass_train.csv"
# TEST_FILE = "../NewDatasets/multiclass_test.csv"
TRAIN_FILE = "../NewDatasets/SMOTEtrain.csv"
TEST_FILE = "../NewDatasets/SMOTEtest.csv"
# TRAIN_FILE = "../NewDatasets/adasyntrain.csv"
# TEST_FILE = "../NewDatasets/adasyntest.csv"
# TRAIN_FILE = "../NewDatasets/border_SMOTEtrain.csv"
# TEST_FILE = "../NewDatasets/border_SMOTEtest.csv"
# TRAIN_FILE = "../NewDatasets/smotetomektrain.csv"
# TEST_FILE = "../NewDatasets/smotetomektest.csv"
# TRAIN_FILE = "../NewDatasets/border_smotetomektrain.csv"
# TEST_FILE = "../NewDatasets/border_smotetomektest.csv"
# TRAIN_FILE = "../NewDatasets/adasyntomektrain.csv"
# TEST_FILE = "../NewDatasets/adasyntomektest.csv"
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)


y_train = train_df['label']
x_train = train_df.drop('label', axis=1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

# Convert features and labels to PyTorch tensors
data_tensor = torch.tensor(x_train.values, dtype=torch.float32)
label_tensor = torch.tensor(y_train.values, dtype=torch.long)

val_data_tensor = torch.tensor(x_val.values, dtype=torch.float32)
val_label_tensor = torch.tensor(y_val.values, dtype=torch.long)

# Define a DataLoader for your training data
train_dataset = torch.utils.data.TensorDataset(data_tensor, label_tensor)
val_dataset = torch.utils.data.TensorDataset(val_data_tensor, val_label_tensor)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
train_set = []
for i, l in train_loader:
    train_set.append((i, l))

y_test = test_df['label']
x_test = test_df.drop('label', axis=1)

# Convert features and labels to PyTorch tensors
data_tensor_test = torch.tensor(x_test.values, dtype=torch.float32)
label_tensor_test = torch.tensor(y_test.values, dtype=torch.long)

# Define a DataLoader for your training data
test_dataset = torch.utils.data.TensorDataset(data_tensor_test, label_tensor_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
test_set = []
for i, l in test_loader:
    test_set.append((i, l))


print(sorted(Counter(y_train).items()))
print(sorted(Counter(y_val).items()))
print(sorted(Counter(y_test).items()))


class_counts = dict(Counter(y_train))
class_counts_sorted = dict(sorted(class_counts.items()))
print(class_counts_sorted)
total_samples = len(y_train)
class_weights = [total_samples/class_counts_sorted[class_label] for class_label in class_counts_sorted]
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)


# Check if a GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

""" Model Training"""
val_accuracy_values = []  # To store val_accuracy values for each epoch
val_losses = []
train_accuracy_values = []  # To store train_accuracy values for each epoch
train_losses = []
test_accuracy_values = []  # To store train_accuracy values for each epoch
test_losses = []
# Initialize the neural network
model = FeedForwardNNComplete(input_size=input_size, hidden_sizes=hidden_size, output_size=num_classes)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-05)

best_accuracy = 0


""" Model Training"""
total_time = 0
total_step = len(train_loader)
for epoch in range(num_epochs):
    start_time = time.time()
    correct_train = 0
    total_n_labels_train = 0
    total_loss_epoch = 0
    for batch_idx, (inputs, labels) in enumerate(train_set):
        # Write the input data and labels to the current device
        inputs, labels = inputs.to(device), labels.to(device)
        # Reset the gradients to zero in the optimizer
        optimizer.zero_grad()
        # Obtain outputs
        logits = model.forward(inputs)
        # print(logits)
        # Obtain predictions
        _, predictions = logits.max(1)
        # Identify how many of the predictions were correct
        correct_train += predictions.eq(labels).sum().item()
        # Add current label count to the total number of labels
        total_n_labels_train += len(labels)
        # Compute loss
        loss = criterion(logits, labels)
        total_loss_epoch += loss.item()
        # Back Propagate the loss
        loss.backward()
        # Apply optimizer
        optimizer.step()
        if batch_idx % log_steps == 0:
            # Calculate the training accuracy
            acc = correct_train / total_n_labels_train
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx}], Loss: {loss.item():.4f},'
                  f' Accuracy: [{acc:.4f}]')
    ep_loss = total_loss_epoch/(batch_idx+1)
    train_losses.append(ep_loss)
    end_time = time.time()
    training_time = end_time - start_time
    total_time += training_time

    """ Model Validation during training to save best model for test evaluation """
    correct = 0
    total = 0
    with torch.no_grad():
        outputs = model(val_data_tensor)
        _, predicted = torch.max(outputs.data, 1)
        total = val_label_tensor.size(0)
        correct = (predicted == val_label_tensor).sum().item()
        accuracy = 100 * correct / total
        val_accuracy_values.append(accuracy)
        loss = criterion(outputs, val_label_tensor)
        val_losses.append(loss.item())
        print(f'Accuracy on validation dataset: {accuracy:.2f}%')

    """ Test evaluation """
    correct = 0
    total = 0
    with torch.no_grad():
        outputs = model(data_tensor_test)
        _, predicted = torch.max(outputs.data, 1)
        total = label_tensor_test.size(0)
        correct = (predicted == label_tensor_test).sum().item()
        accuracy = 100 * correct / total
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            if os.path.exists('best_model_parameters.pt'):
                os.remove('best_model_parameters.pt')
            torch.save(model.state_dict(), 'best_model_parameters.pt')
        test_accuracy_values.append(accuracy)
        loss = criterion(outputs, label_tensor_test)
        test_losses.append(loss.item())
        print(f'Accuracy on test dataset: {accuracy:.2f}%')


print("Total Training Time: ", total_time)
""" Final Validation Set Model Evaluation """
print("============================= Validation Set ==========================================")
class_labels = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
correct = 0
total = 0
outputs = model(val_data_tensor)
_, predicted = torch.max(outputs.data, 1)
y_pred = predicted.numpy()
y_true = val_label_tensor.numpy()
matrix, report_imb, accuracy = perf_evaluation(y_true, y_pred, class_labels, True)

""" Test Set Model Evaluation """
print("============================= Test Set ==========================================")
best_model = FeedForwardNNComplete(input_size=input_size, hidden_sizes=hidden_size, output_size=num_classes)
best_model.load_state_dict(torch.load('best_model_parameters.pt'))
class_labels = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
correct = 0
total = 0
outputs = best_model(data_tensor_test)
_, predicted = torch.max(outputs.data, 1)
y_pred = predicted.numpy()
y_true = label_tensor_test.numpy()
matrix, report_imb, accuracy = perf_evaluation(y_true, y_pred, class_labels, True)

print("Train Losses through epochs:")
print(train_losses)
print("Val Losses through epochs:")
print(val_losses)
print("Val Accuracies through epochs")
print(val_accuracy_values)
print("Test Losses through epochs:")
print(test_losses)
print("Test Accuracies through epochs")
print(test_accuracy_values)


"""Plot loss over epochs"""
plt.plot(range(1, num_epochs+1), val_losses, marker='o', linestyle='-')
plt.plot(range(1, num_epochs+1), train_losses, marker='o', linestyle='-')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.show()


