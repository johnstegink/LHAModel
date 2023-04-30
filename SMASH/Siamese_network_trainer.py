# Class to train the siamese network

import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader

from SMASH.Siamese import Siamese
import torch.optim as optimizer

class SiameseNetworkTrainer:

    def __init__(self):
        self.dummy = 1


    def train(self, preprocessor,  nr_of_epochs):
        device = preprocessor.get_device()
        model = Siamese(preprocessor=preprocessor).to( device)
        opt = optimizer.SGD(model.parameters(), lr=0.01, momentum=0.9)  # Gradient descent optimizer
        loss_fn = nn.BCEWithLogitsLoss()

        # get the datasets
        dataset = preprocessor.create_dataset()
        train_dataset, val_dataset = random_split( dataset, [80,20])
        train_loader = DataLoader(dataset=train_dataset, batch_size=16)
        val_loader = DataLoader(dataset=val_dataset, batch_size=20)

        losses = []
        val_losses = []

        running_loss = 0.0
        model.train()
        for epoch in range( nr_of_epochs):
            for x1_batch, x2_batch, y_batch in train_loader:
                x1_batch = x1_batch.to(device)
                x2_batch = x2_batch.to(device)
                y_batch = y_batch.to(device)

                yhat = model(x1_batch, x2_batch)
                loss = loss_fn( yhat, y_batch)
                loss.backward()
                opt.step()
                opt.zero_grad()

                losses.append(loss)

            for x1_batch, x2_batch, y_batch in train_loader:
                with torch.no_grad():
                    for x1_val, x2_val, y_val in val_loader:
                        x1_val = x1_val.to(device)
                        x2_val = x2_val.to(device)
                        y_val = y_val.to(device)

                        model.eval()

                        yhat = model(x1_val, x2_val)
                        val_loss = loss_fn(y_val, yhat)
                        val_losses.append(val_loss.item())



                # running_loss += loss.item()
                # if i % 20 == 0:    # print every 20 mini-batches
                #     print('[%d, %5d] loss: %.3f' %
                #           (epoch + 1, i + 1, running_loss / 20))
                #     running_loss = 0.0
