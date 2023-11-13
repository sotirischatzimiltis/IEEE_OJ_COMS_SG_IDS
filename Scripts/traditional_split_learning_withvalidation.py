from mpi4py import MPI

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from time import time
import numpy as np
from sys import argv
from argparse import ArgumentParser, Namespace
from models import FeedForwardNNClient, FeedForwardNNServer, perf_evaluation
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import Counter


def parse_args() -> Namespace:
    """Parses CL arguments

    Returns:
        Namespace object containing all arguments
    """
    parser = ArgumentParser()

    parser.add_argument("-bs", "--batch_size", type=int, default=1024)
    parser.add_argument("-nb", "--num_batches", type=int, default=98)
    parser.add_argument("-tbs", "--test_batch_size", type=int, default=1000)
    parser.add_argument("-ls", "--log_steps", type=int, default=20)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01)
    parser.add_argument("-e", "--epochs", type=int, default=20)
    parser.add_argument("-p", "--plot", type=bool, default=False)

    return parser.parse_args(argv[1:])


# Get the assigned arguments
args = parse_args()

"""Data Manipulation"""
# TRAIN_FILE = "../Datasets/train_multiclass.csv"
# TEST_FILE = "../Datasets/test_multiclass.csv"

TRAIN_FILE = "../Datasets/smotetomektrain.csv"
TEST_FILE = "../Datasets/smotetomektest.csv"

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

y_train = train_df['label']
x_train = train_df.drop('label', axis=1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)

# Convert features and labels to PyTorch tensors
data_tensor = torch.tensor(x_train, dtype=torch.float32)
label_tensor = torch.tensor(y_train.values, dtype=torch.long)

val_data_tensor = torch.tensor(x_val, dtype=torch.float32)
val_label_tensor = torch.tensor(y_val.values, dtype=torch.long)

# Define a DataLoader for your training data
train_dataset = torch.utils.data.TensorDataset(data_tensor, label_tensor)
val_dataset = torch.utils.data.TensorDataset(val_data_tensor, val_label_tensor)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

train_set = []
for i, l in train_loader:
    train_set.append((i, l))

y_test = test_df['label']
x_test = test_df.drop('label', axis=1)
x_test = scaler.transform(x_test)
# Convert features and labels to PyTorch tensors
data_tensor_test = torch.tensor(x_test, dtype=torch.float32)
label_tensor_test = torch.tensor(y_test.values, dtype=torch.long)

# Define a DataLoader for your training data
test_dataset = torch.utils.data.TensorDataset(data_tensor_test, label_tensor_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
test_set = []
for i, l in test_loader:
    test_set.append((i, l))


# Define the communicator
comm = MPI.COMM_WORLD
# Get the rank of the current process in the communicator
rank = comm.Get_rank()

# The rank of the server is 0
SERVER = 0
# MAX_RANK is the highest rank value of a process
MAX_RANK = comm.Get_size() - 1
print(MAX_RANK)
# Set a random seed
torch.manual_seed(0)

# Check if a GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Check if the current process is a worker or the server
# If rank is 0 then it's the server else is a worker
if rank >= 1:
    worker, server = True, False
else:
    server, worker = True, False

# assign worker names
worker_map = {1: "alfa", 2: "bravo", 3: "charlie", 4: "delta", 5: "echo", 6: "foxtrot", 7: "golf", 8: "hotel",
              9: "india", 10: "juliet"}

# Define the procedure for the workers
if worker:
    epoch = 1
    active_worker = rank
    worker_left = rank - 1
    worker_right = rank + 1

    # The server is not a worker
    # If am at the first process then the worker in the left is the last
    if rank == 1:
        worker_left = MAX_RANK
        print(sorted(Counter(y_train).items()))
        print(sorted(Counter(y_val).items()))
        print(sorted(Counter(y_test).items()))
    # Make sure that worker rank 1 is the first to start
    # If am at the last process then the worker in the right is the first
    elif rank == MAX_RANK:
        worker_right = 1
        comm.send("you_can_start", dest=worker_right)
        print("RANK MAX RANK")

    # Make sure that each worker has its own private training data
    # Give different data chunk to every client
    start = int(np.floor(len(train_set) / MAX_RANK * (rank - 1)))
    stop = int(np.floor(len(train_set) / MAX_RANK * rank))
    worker_train_set = train_set[start:stop]

    # Instantiate the client network
    model = FeedForwardNNClient().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    while True:
        # Wait to receive a message from the other worker
        msg = comm.recv(source=worker_left)
        split_layer_tensor = grads = inputs = labels = 0

        if msg == "you_can_start":
            # This means that we are either in epoch 1 or we did a full loop
            if rank == 1:
                print(f"\nStart epoch {epoch}:")

            start = time()
            for batch_idx, (inputs, labels) in enumerate(worker_train_set):
                # Write the input data and labels to the current device
                inputs, labels = inputs.to(device), labels.to(device)

                # Reset the gradients to zero in the optimizer
                optimizer.zero_grad()

                # Obtain the tensor output of the split layer
                split_layer_tensor = model(inputs)

                # Send tensor of the split layer and the labels to the server
                comm.send(["tensor_and_labels", [split_layer_tensor, labels]], dest=SERVER)

                # Receive the gradients for back-propagation from the server
                grads = comm.recv(source=SERVER)

                # Apply the gradients to the split layer tensor
                split_layer_tensor.backward(grads)

                # Apply the optimizer
                optimizer.step()

            # Garbage collection
            del split_layer_tensor, grads, inputs, labels
            torch.cuda.empty_cache()
            end = time()

            # Send training time to server
            comm.send(["time", end-start], dest=SERVER)

            # Only let the last worker evaluate on the test set
            if rank == MAX_RANK:
                # Tell the server to start validating
                comm.send("validation", dest=SERVER)
                for batch_idx, (inputs, labels) in enumerate(val_loader):
                    # Write the input data and labels to the current device
                    inputs, labels = inputs.to(device), labels.to(device)

                    # Obtain the tensor output of the split layer
                    split_layer_tensor = model(inputs)

                    # Send tensor of the split layer and the labels to the server
                    comm.send(["tensor_and_labels", [split_layer_tensor, labels]], dest=SERVER)

                comm.send("testing", dest=SERVER)
                for batch_idx, (inputs, labels) in enumerate(test_loader):
                    # Write the input data and labels to the current device
                    inputs, labels = inputs.to(device), labels.to(device)

                    # Obtain the tensor output of the split layer
                    split_layer_tensor = model(inputs)

                    # Send tensor of the split layer and the labels to the server
                    comm.send(["tensor_and_labels", [split_layer_tensor, labels]], dest=SERVER)

                # Garbage collection
                del split_layer_tensor, inputs, labels
                torch.cuda.empty_cache()

            if rank == MAX_RANK and epoch == args.epochs:
                comm.send("final_testing", dest=SERVER)

                val_split_layer_tensor = model(val_data_tensor)
                test_split_layer_tensor = model(data_tensor_test)
                comm.send(["tensor_and_labels", [val_split_layer_tensor, val_label_tensor, test_split_layer_tensor,
                                                 label_tensor_test]],
                          dest=SERVER)

            # Signal to the other worker that it can start training
            comm.send("you_can_start", dest=worker_right)

            if epoch == args.epochs:
                msg = "training_complete" if rank == MAX_RANK else "worker_done"
                comm.send(msg, dest=SERVER)
                exit()
            else:
                # Let the server know that the current epoch has finished
                msg = "epoch_done" if rank == MAX_RANK else "worker_done"
                comm.send(msg, dest=SERVER)

            epoch += 1

# Define the procedure for the server
elif server:
    print("server here")
    # Instantiate the server network
    model = FeedForwardNNServer()
    model = model.to(device)

    # Define the loss criterion
    loss_crit = nn.CrossEntropyLoss()

    # Use Stochastic Gradient Descent with momentum and weight decay
    # as the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    total_training_time = 0.0

    epoch, step, batch_idx = 1, 0, 0
    test_step, val_step, train_step = 0, 0, 0

    active_worker, phase = 1, "train"

    val_loss, val_losses, val_accs = 0.0, [], []
    test_loss, test_losses, test_accs = 0.0, [], []
    train_loss, train_losses = 0, []
    total_n_labels_train, total_n_labels_test, total_n_labels_val = 0, 0, 0
    correct_train, correct_test, correct_val = 0, 0, 0

    while True:
        # Wait for the message of the active worker
        msg = comm.recv(source=active_worker)

        if msg[0] == "tensor_and_labels":
            if phase == "final_testing":
                val_split_layer_tensor, val_label_tensor, test_split_layer_tensor, label_tensor_test = msg[1]
                logits_val = model(val_split_layer_tensor)
                logits_test = model(test_split_layer_tensor)
                _, predictions_val = logits_val.max(1)
                _, predictions_test = logits_test.max(1)
                y_pred_val = predictions_val.numpy()
                y_true_val = val_label_tensor.numpy()
                y_pred_test = predictions_test.numpy()
                y_true_test = label_tensor_test.numpy()
                class_labels = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
                perf_evaluation(y_true_val, y_pred_val, class_labels, name="validation", plot=True)
                perf_evaluation(y_true_test, y_pred_test, class_labels, name="test", plot=True)
                print("Total training time: {:.2f}s".format(total_training_time))
                print("Train Losses through epochs:")
                print(train_losses)
                print("Val Losses through epochs:")
                print(val_losses)
                print("Val Accuracies through epochs")
                print(val_accs)
                print("Test Losses through epochs:")
                print(test_losses)
                print("Test Accuracies through epochs")
                print(test_accs)
                exit()
            if phase == "train":
                # Reset the gradients to zero in the optimizer
                optimizer.zero_grad()

            # Dereference the input tensor and corresponding labels from the
            # message
            input_tensor, labels = msg[1]
            # print(labels)
            # Obtain logits from forward pass through server model
            logits = model(input_tensor)
            # print(logits)

            # Obtain the predictions
            _, predictions = logits.max(1)

            # Compute the loss
            loss = loss_crit(logits, labels)

            if phase == "train":
                train_step += 1
                train_loss += loss.item()
                # Add current label count to the total number of labels
                total_n_labels_train += len(labels)

                # Identify how many of the predictions were correct
                correct_train += predictions.eq(labels).sum().item()

                # Back-propagate the loss
                loss.backward()

                # Apply the optimizer
                optimizer.step()

                # Send gradients back to the active worker
                comm.send(input_tensor.grad, dest=active_worker)

                # Increment batch index
                batch_idx += 1

                if batch_idx % args.log_steps == 0:
                    # Calculate the training accuracy
                    acc = correct_train / total_n_labels_train

                    print('{} - Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                        worker_map[active_worker], epoch,
                        int((args.num_batches * args.batch_size) / MAX_RANK * (active_worker-1)) +
                        batch_idx * args.batch_size, args.num_batches * args.batch_size,
                        100. * (((args.num_batches / MAX_RANK * (active_worker-1)) + batch_idx) / args.num_batches),
                        loss.item()))

            if phase == "validation":
                val_step += 1
                # Add current label count to the total number of labels
                total_n_labels_val += len(labels)

                # Identify how many of the predictions were correct
                correct_val += predictions.eq(labels).sum().item()
                val_loss += loss.item()

            if phase == "testing":
                test_step += 1
                test_loss += loss.item()
                # Add current label count to the total number of labels
                total_n_labels_test += len(labels)

                # Identify how many of the predictions were correct
                correct_test += predictions.eq(labels).sum().item()

        elif msg[0] == "time":
            total_training_time += msg[1]

        elif msg == "worker_done":
            if active_worker == MAX_RANK:
                epoch += 1

            # Change worker and phase
            active_worker = (active_worker % MAX_RANK) + 1
            phase = "train"

            # Reset variables
            total_n_labels_train, correct_train, batch_idx = 0, 0, 0

        elif msg == "epoch_done" or msg == "training_complete":
            # Update the validation loss
            val_loss /= val_step
            val_losses.append(val_loss)
            val_acc = correct_val / total_n_labels_val
            val_accs.append(val_acc)

            test_loss /= test_step
            test_losses.append(test_loss)
            test_acc = correct_test / total_n_labels_test
            test_accs.append(test_acc)

            train_loss /= (train_step*MAX_RANK)
            train_losses.append(train_loss)

            print("\nVal set - Epoch: {} - Loss: {:.4f}, Acc: ({:2f}%)\n".format(
                epoch, val_loss, 100 * val_acc))

            print("\nTest set - Epoch: {}, Acc: ({:2f}%)\n".format(
                epoch, 100 * test_acc))

            if active_worker == MAX_RANK:
                epoch += 1

            # Change worker and phase
            active_worker = (active_worker % MAX_RANK) + 1
            phase = "train"

            # Reset variables
            total_n_labels_test, correct_test = 0, 0
            total_n_labels_val, correct_val = 0, 0
            test_step, val_step, train_step, batch_idx = 0, 0, 0, 0

            if msg == "training_complete":
                print("Training complete.")
                # Create validation loss and accuracy plots
                epoch_list = list(range(1, args.epochs + 1))
                print("Total training time: {:.2f}s".format(total_training_time))
                print("Final test accuracy: {:.4f}".format(acc))
                print("Final test loss: {:.4f}".format(val_loss))
                exit()
            # Reset loss if training not complete
            val_loss, train_loss, test_loss = 0.0, 0.0, 0.0 
        elif msg == "validation":
            # Change phase and reset variables
            phase = "validation"
            step, total_n_labels_train, correct_train = 0, 0, 0

        elif msg == "testing":
            test_step = 0
            phase = "testing"

        elif msg == "final_testing":
            phase = "final_testing"
