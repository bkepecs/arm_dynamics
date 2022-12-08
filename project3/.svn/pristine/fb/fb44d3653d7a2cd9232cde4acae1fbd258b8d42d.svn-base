import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import torch.optim as optim
import argparse
import time
from models import *
np.set_printoptions(suppress=True)


class DynamicDataset(Dataset):
    def __init__(self, dataset_dir):
        # X: (N, 9), Y: (N, 6)
        self.X = np.load(os.path.join(dataset_dir, 'X.npy')).T.astype(np.float32)
        self.Y = np.load(os.path.join(dataset_dir, 'Y.npy')).T.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def train(model, dataloader, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X,y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        # Compute loss between prediction and ground truth
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Report statistics
        if batch % 100 == 0:
            loss = loss.item()
            current = batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(model, dataloader, loss_fn, device):
    model.eval()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            #print("Abs:")
            #print(np.abs(pred.item()-y.item()))
            correct += np.linalg.norm(pred-y)
    test_loss /= num_batches
    print(f"Test Error: \n Norm: {(100*correct):>0.1f}, Avg loss: {test_loss:>8f} \n")

    return test_loss


def get_args():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--num_links', type=int, default=3)
    parser.add_argument('--split', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--save_dir', type=str, default='models')
    args = parser.parse_args()
    args.timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    args.save_dir = os.path.join(args.save_dir, args.timestr)
    return args


def main():
    # Specify device
    args = get_args()
    use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using {device} device")
    
    # Specify parameters
    batch_size = 10000
    time_step = 0.01
    learning_rate = .0001
    num_epochs = 2000
    
    # Load and split dataset
    dataset = DynamicDataset(args.dataset_dir)
    dataset_size = len(dataset)
    test_size = int(np.floor(args.split * dataset_size))
    train_size = dataset_size - test_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    # Initialize the model
    model = build_model(args.num_links, time_step).to(device)
    # see if we can keep training
    #model.load_state_dict(torch.load("3linkarm/models/2022-04-11_09-04-43/epoch_0221_loss_0.00015066/dynamics.pth"))
    
    # Specify optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    # Train the model
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(model, train_loader, loss_fn, optimizer, device)
        test_loss = test(model, test_loader, loss_fn, device)
        model_folder_name = f'epoch_{epoch:04d}_loss_{test_loss:.8f}'

        # Save the model at this epoch
        if not os.path.exists(os.path.join(args.save_dir, model_folder_name)):
            os.makedirs(os.path.join(args.save_dir, model_folder_name))
        torch.save(model.state_dict(), os.path.join(args.save_dir, model_folder_name, 'dynamics.pth'))
        print(f'model saved to {os.path.join(args.save_dir, model_folder_name, "dynamics.pth")}\n')

    print("Done!")


if __name__ == '__main__':
    main()
