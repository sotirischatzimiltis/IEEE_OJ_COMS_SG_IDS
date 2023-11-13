from mpi4py import MPI
import torch
import torch.nn as nn
import pandas as pd
from time import time
from sys import argv
from argparse import ArgumentParser, Namespace
from models import FeedForwardNNClientNols, FeedForwardNNServerNols, perf_evaluation
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from sklearn.model_selection import train_test_split


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


args = parse_args()  # Get the assigned arguments

"""Data Manipulation"""
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
data_tensor = torch.tensor(x_train.values, dtype=torch.float32)
label_tensor = torch.tensor(y_train.values, dtype=torch.long)

val_data_tensor = torch.tensor(x_val.values, dtype=torch.float32)
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
data_tensor_test = torch.tensor(x_test.values, dtype=torch.float32)
label_tensor_test = torch.tensor(y_test.values, dtype=torch.long)

# Define a DataLoader for your training data
test_dataset = torch.utils.data.TensorDataset(data_tensor_test, label_tensor_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
test_set = []
for i, l in test_loader:
    test_set.append((i, l))

class_counts = dict(Counter(y_train))
class_counts_sorted = dict(sorted(class_counts.items()))
print(class_counts_sorted)
total_samples = len(y_train)
class_weights = [total_samples/class_counts_sorted[class_label] for class_label in class_counts_sorted]
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)


comm = MPI.COMM_WORLD  # Define the communicator
rank = comm.Get_rank()  # Get the rank of the current process in the communicator
SERVER = 0  # The rank of the server it 0
MAX_RANK = comm.Get_size() - 1  # MAX_RANK is the highest rank value of a process

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


if worker:
    epoch = 1
    active_worker = rank

    test_loss, val_loss, train_loss = 0.0, 0.0, 0.0
    total_n_labels_test, total_n_labels_val, total_n_labels_train = 0, 0, 0
    correct_test, correct_val, correct_train = 0, 0, 0

    model = FeedForwardNNClientNols().to(device)  # Instantiate the client network
    loss_crit = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08,
                                 weight_decay=1e-05)

    print(sorted(Counter(y_train).items()))
    print(sorted(Counter(y_val).items()))
    print(sorted(Counter(y_test).items()))

    while True:
        split_layer_tensor = grads = inputs = labels = 0
        print(f"\nStart epoch {epoch}:")

        start = time()
        for batch_idx, (inputs, labels) in enumerate(train_set):
            inputs, labels = inputs.to(device), labels.to(device)  # Write the input data and labels to device
            optimizer.zero_grad()  # Reset the gradients to zero in the optimizer
            split_layer_tensor = model.forward(inputs)  # Obtain the split layer  tensor output
            comm.send(["tensor", split_layer_tensor], dest=SERVER)  # Send tensor and labels to the server
            server_outputs = comm.recv(source=SERVER)  # Receive the outputs of the 3rd hidden layer
            logits = model.final_layer_forward(server_outputs)  # Obtain outputs
            _, predictions = logits.max(1)  # Obtain predictions
            correct_train += predictions.eq(labels).sum().item()   # Identify how many of the predictions were correct
            total_n_labels_train += len(labels)  # Add current label count to the total number of labels
            loss = loss_crit(logits, labels)   # Compute loss
            loss.backward()  # Back Propagate the loss
            optimizer.step()  # Apply optimizer
            comm.send(server_outputs.grad, dest=SERVER)   # Send to the server the gradient from output layer
            grads = comm.recv(source=SERVER)  # Receive gradients for backpropagation from the server
            split_layer_tensor.backward(grads)   # Apply the gradients to the split layer tensor
            optimizer.step()   # Apply optimizer
            comm.send(["metrics", [correct_train, total_n_labels_train, loss]], dest=SERVER)  # Send metrics to server

        # Garbage collection
        del split_layer_tensor, grads, inputs, labels
        torch.cuda.empty_cache()
        end = time()

        comm.send(["time", end-start], dest=SERVER)  # Send training time to server

        # Get validation performance after training epoch
        comm.send("validation", dest=SERVER)
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            split_layer_tensor = model.forward(inputs)
            comm.send(["tensor", split_layer_tensor], dest=SERVER)
            server_outputs = comm.recv(source=SERVER)
            logits = model.final_layer_forward(server_outputs)
            _, predictions = logits.max(1)
            loss = loss_crit(logits, labels)
            total_n_labels_val += len(labels)
            correct_val += predictions.eq(labels).sum().item()
            val_loss += loss.item()
            comm.send(["metrics", [total_n_labels_val, correct_val, val_loss]], dest=SERVER)  # Send the metrics

        # Get testing performance after training epoch
        comm.send("testing", dest=SERVER)
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            split_layer_tensor = model.forward(inputs)
            comm.send(["tensor", split_layer_tensor], dest=SERVER)
            server_outputs = comm.recv(source=SERVER)
            logits = model.final_layer_forward(server_outputs)
            _, predictions = logits.max(1)
            loss = loss_crit(logits, labels)
            total_n_labels_test += len(labels)
            correct_test += predictions.eq(labels).sum().item()
            test_loss += loss.item()
            comm.send(["metrics", [total_n_labels_test, correct_test, test_loss]], dest=SERVER)  # Send the metrics

        # Garbage collection
        del split_layer_tensor, inputs, labels
        torch.cuda.empty_cache()

        if epoch == args.epochs:
            comm.send('final_testing', dest=SERVER)

            """ Val set evaluation"""
            val_split_layer_tensor = model.forward(val_data_tensor)
            comm.send(["tensor", val_split_layer_tensor], dest=SERVER)
            fc = comm.recv(source=SERVER)
            logits = model.final_layer_forward(fc)
            _, predictions_val = logits.max(1)
            y_pred_val = predictions_val.numpy()
            y_true_val = val_label_tensor.numpy()
            comm.send(["val_set", [y_pred_val, y_true_val]], dest=SERVER)

            """ Test set evaluation """
            test_split_layer_tensor = model.forward(data_tensor_test)
            comm.send(["tensor", test_split_layer_tensor], dest=SERVER)
            fc = comm.recv(source=SERVER)
            logits = model.final_layer_forward(fc)
            _, predictions_test = logits.max(1)
            y_pred_test = predictions_test.numpy()
            y_true_test = label_tensor_test.numpy()
            comm.send(["test_set", [y_pred_test, y_true_test]], dest=SERVER)
            exit()

        batch_idx = 0
        test_loss, val_loss, train_loss = 0.0, 0.0, 0.0
        total_n_labels_test, total_n_labels_val, total_n_labels_train = 0, 0, 0
        correct_test, correct_val, correct_train = 0, 0, 0
        comm.send("epoch_done", dest=SERVER)
        epoch += 1

elif server:
    # Instantiate the server network
    model = FeedForwardNNServerNols().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08,
                                 weight_decay=1e-05)

    total_training_time = 0.0
    epoch, step, batch_idx = 1, 0, 0
    active_worker, phase = 1, "train"
    val_loss, val_losses, val_accs = 0.0, [], []
    test_loss, test_losses, test_accs = 0.0, [], []
    train_loss, train_losses = 0.0, []
    total_n_labels_train, total_n_labels_test, total_n_labels_val = 0, 0, 0
    correct_train, correct_test, correct_val = 0, 0, 0
    train_step, val_step, test_step = 0, 0, 0

    while True:
        # Wait for the message of the active worker
        msg = comm.recv(source=active_worker)
        if msg[0] == "tensor":
            if phase == "final_testing":
                print("Final Testing")
                # Val Set Evaluation
                input_tensor = msg[1]  # Dereference the tensor from the split layer
                output = model.forward(input_tensor)
                comm.send(output, dest=active_worker)  # Send output to the user to finalize forward pass
                msg = comm.recv(source=active_worker)  # Receive prediction
                y_pred_val, y_true_val = msg[1]

                # Test Set Evaluation
                msg = comm.recv(source=active_worker)  # Dereference the tensor from the split layer
                input_tensor = msg[1]
                output = model.forward(input_tensor)  # Obtain outputs from hidden layer forward pass
                comm.send(output, dest=active_worker)  # Send output to the user to finalize pass
                msg = comm.recv(source=active_worker)
                y_pred_test, y_true_test = msg[1]

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

            input_tensor = msg[1]  # Dereference the input tensor
            output = model.forward(input_tensor)  # Obtain outputs from first hidden layer forward pass
            comm.send(output, dest=active_worker)  # Send output to the user to finalize the training

            if phase == "train":
                train_step += 1
                grad = comm.recv(source=active_worker)  # Receive gradient from output layer
                output.backward(grad)  # Back Propagate the gradients
                optimizer.step()  # Apply the optimizer
                comm.send(input_tensor.grad, dest=active_worker)  # Send gradients back to the active worker
                batch_idx += 1  # Increment batch index

                message = comm.recv(source=active_worker)
                correct_train, total_n_labels_train, loss = message[1]
                train_loss += loss.item()
                if batch_idx % args.log_steps == 0:
                    acc = correct_train / total_n_labels_train  # Calculate the training accuracy
                    print('{} - Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                        worker_map[active_worker], epoch,
                        int((args.num_batches * args.batch_size) / MAX_RANK * (active_worker-1)) +
                        batch_idx * args.batch_size, args.num_batches * args.batch_size,
                        100. * (((args.num_batches / MAX_RANK * (active_worker-1)) + batch_idx) / args.num_batches),
                        loss.item()))

            if phase == "validation":
                val_step += 1
                message = comm.recv(source=active_worker)
                labels_val, c_val, b_loss = message[1]
                total_n_labels_val += labels_val
                correct_val += c_val
                val_loss += b_loss

            if phase == "testing":
                test_step += 1
                message = comm.recv(source=active_worker)
                labels_test, c_test, b_loss = message[1]
                total_n_labels_test += labels_test
                correct_test += c_test
                test_loss += b_loss

        elif msg[0] == "time":
            total_training_time += msg[1]

        elif msg == "epoch_done":
            phase = "train"
            # Update validation loss
            val_loss /= val_step*MAX_RANK
            print("Epoch val loss:", val_loss)
            val_losses.append(val_loss)
            val_acc = correct_val / total_n_labels_val
            val_accs.append(val_acc)
            # Update test loss
            test_loss /= test_step*MAX_RANK
            test_losses.append(test_loss)
            test_acc = correct_test / total_n_labels_test
            test_accs.append(test_acc)
            # Update train loss
            train_loss /= (train_step * MAX_RANK)
            train_losses.append(train_loss)

            print("\nVal set - Epoch: {} - Loss: {:.4f}, Acc: ({:2f}%)\n".format(
                epoch, val_loss, 100 * val_acc))

            print("\nTest set - Epoch: {}, Acc: ({:2f}%)\n".format(
                epoch, 100 * test_acc))

            epoch += 1

            # Reset variables
            total_n_labels_test, correct_test = 0, 0
            total_n_labels_val, correct_val = 0, 0
            test_step, val_step, train_step, batch_idx = 0, 0, 0, 0
            total_n_labels_train, correct_train = 0, 0

            # Reset loss if training not complete
            val_loss, train_loss, test_loss = 0.0, 0.0, 0.0

        elif msg == "validation":
            phase = "validation"
        elif msg == "testing":
            phase = "testing"
        elif msg == "final_testing":
            phase = "final_testing"

