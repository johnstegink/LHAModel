# Class to train the siamese network

import torch
import torch.nn as nn
from SMASH.Siamese import Siamese
import torch.optim as optimizer

class SiameseNetworkTrainer:

    def __init__(self):
        self.dummy = 1


    def train(self, preprocessor,  nr_of_epochs):
        dev = preprocessor.get_device()
        model = Siamese(preprocessor=preprocessor)
        opt = optimizer.SGD(model.parameters(), lr=0.01, momentum=0.9)  # Gradient descent optimizer
        loss_fn = nn.BCEWithLogitsLoss()
        model.to( dev)


        running_loss = 0.0
        for epoch in range( nr_of_epochs):
            for i, data in enumerate( preprocessor.create_training_dataset()):
                (X1, X2, target) = data

                opt.zero_grad()

                outputs = model(X1, X2)
                loss = loss_fn( outputs, target)
                loss.backward()
                opt.step()

                running_loss += loss.item()
                if i % 20 == 0:    # print every 20 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
