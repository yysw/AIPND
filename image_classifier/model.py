#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

class myModel:

    def __init__(self, arch, hidden_units, gpu, pretrained):
        self.arch = arch
        self.hidden_units = hidden_units
        self.pretrained = pretrained
        self.gpu = gpu

    def load_pretrained_model(self):
        arch = self.arch
        pretrained = self.pretrained
        if arch == 'vgg11':
            model = models.vgg11(pretrained)
        elif  arch == 'vgg11_bn':
            model = models.vgg11_bn(pretrained)
        elif  arch == 'vgg13':
            model = models.vgg13(pretrained)
        elif  arch == 'vgg13_bn':
            model = models.vgg13_bn(pretrained)
        elif  arch == 'vgg16':
            model = models.vgg16(pretrained)
        elif  arch == 'vgg16_bn':
            model = models.vgg16_bn(pretrained)
        elif  arch == 'vgg19':
            model = models.vgg19(pretrained)
        elif  arch == 'vgg19_bn':
            model = models.vgg19_bn(pretrained)
        return model


    # Redefine the classifier
    def redefine_model(self, model):
        # Freeze parameters
        for param in model.parameters():
            param.requires_grad = False

        # Define feed-forward network
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(25088, self.hidden_units)),
                                  ('relu', nn.ReLU()),
                                  ('fc2', nn.Linear(self.hidden_units, 102)),
                                  ('output', nn.LogSoftmax(dim=1))
                                  ]))

        model.classifier = classifier
        return model

    # Train the feed-forward network
    def train_model(self, model, trainloader, validloader, learning_rate, epochs=1, print_every=5):
        steps = 0
        running_loss = 0
        criterion = nn.NLLLoss()

        device = torch.device("cuda" if self.gpu else "cpu")
        print("train with {}".format("gpu" if self.gpu else "cpu"))
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
        model.to(device);

        for epoch in range(epochs):
            for inputs, labels in trainloader:
                steps += 1
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)

                logps = model.forward(inputs)
                loss = criterion(logps, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    valid_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in validloader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)

                            valid_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                          f"Validation accuracy: {accuracy/len(validloader):.3f}")
                    running_loss = 0
                    model.train()
                
    # Do test on the test set
    def test_model(self, model, testloader):
        test_loss = 0
        accuracy = 0
        criterion = nn.NLLLoss()

        device = torch.device("cuda" if self.gpu else "cpu")
        model.eval()
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)

                test_loss += batch_loss.item()

                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Test loss: {test_loss/len(testloader):.3f}.. "
              f"Test accuracy: {accuracy/len(testloader):.3f}")

    # Save the checkpoint 
    def save_model(self, model, class_to_idx):
        checkpoint = {'class_to_idx': class_to_idx,
                      'state_dict': model.state_dict()}

        torch.save(checkpoint, self.save_dir+'checkpoint.pth')

    # Loads a checkpoint and rebuilds the model
    def load_model(self):
        # Define a model with the same architecture
        model = load_pretrained_model(self.arch)
        model = redefine_model(model, self.hidden_units)
        checkpoint = torch.load(self.save_dir+'checkpoint.pth')
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']

        return model
