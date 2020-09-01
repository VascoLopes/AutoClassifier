import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets

import numpy as np
import copy
import time
import os
from collections import OrderedDict 


class Utils():

    #--------------------------------------------------------------------------------#
    def __init__(self, batch_size, device='cuda'):
    # Parameters:
        self.batch_size   = batch_size
        self.device       = device
        
    #--------------------------------------------------------------------------------#
    def getCrossEntropyLoss(self, weights=None):
        #print("[Using CrossEntropyLoss...]")
        if weights is not None:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.CrossEntropyLoss(weight=weights)

        return (criterion)

    #--------------------------------------------------------------------------------#
    def getSGDOptimizer(self, model, learningRate = 0.001, momentum=0.9):
        #print("[Using small learning rate with momentum...]")
        optimizer_conv = optim.SGD(list(filter(
            lambda p: p.requires_grad, model.parameters())), lr=learningRate, momentum=momentum)

        return (optimizer_conv)

    #--------------------------------------------------------------------------------#
    def getLrScheduler(self, model, step_size=7, gamma=0.1):
        print("[Creating Learning rate scheduler...]")
        exp_lr_scheduler = lr_scheduler.StepLR(model, step_size=step_size, gamma=gamma)

        return (exp_lr_scheduler)

    #--------------------------------------------------------------------------------#
    ''' Train function '''
    def train(self, model, dataloader, criterion, optimizer, epoch=None):
        model.train()

        correct = 0
        total = 0
        correct_batch = 0
        total_batch = 0
        lossTotal = 0
        predictions = [] # Store all predictions, for metric analysis

        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            #print(f'Input shape: {outputs.shape}')
            #print(f'Layer: {layer}')
            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss = loss.item() * inputs.size(0)
            lossTotal += running_loss # epoch loss
            correct_batch = (predicted == labels).sum().item()
            total_batch = labels.size(0)
            predictions +=  list(predicted.cpu().numpy())

            #if i % 200 == 0:    # print every 200 mini-batches
            #    print('[%d, %5d] loss: %.5f ; Accuracy: %.2f'%
            #        (epoch, i + 1, running_loss/total_batch, 100 * correct_batch / total_batch))

            running_loss = 0.0
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = 100 * correct / total
        lossTotal = lossTotal/total
        if epoch != None:
            print(f'Epoch {epoch} - Train Accuracy: {accuracy},    Loss: {lossTotal}')

        return model, accuracy, lossTotal, predictions

    #--------------------------------------------------------------------------------#
    ''' Validation function '''
    def evaluate(self, model, valloader, criterion, epoch=None):
        model.eval()
        correct = 0
        total = 0
        running_loss = 0
        predictions = [] # Store all predictions, for metric analysis

        with torch.no_grad():
            for data in valloader:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                #outputs = model(inputs)

                outputs = model(inputs)

                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                predictions +=  list(predicted.cpu().numpy())

        acc  = 100 * correct / total
        lossTotal = running_loss / total
        if epoch != None:
            print(f'Epoch {epoch} - Val Accuracy: {acc},    Loss: {loss}')
        return acc, lossTotal, predictions

    #--------------------------------------------------------------------------------#
    ''' Test function '''
    def test(self, model, testloader):
        model.eval()
        correct = 0
        total = 0
        predictions = []
        trueLabels = []
        inferenceTime = []
        with torch.no_grad():
            startTime = time.time()
            for data in testloader:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                trueLabels += list(labels.cpu().numpy())
                predictions +=  list(predicted.cpu().numpy())
                inferenceTime.append(time.time()-startTime)

        acc = 100 * correct / total
        return acc, predictions, trueLabels, inferenceTime


    #--------------------------------------------------------------------------------#
    def getCifar10Dataset(self):
        CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
        CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

        transform_train = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

        validation_test = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train='train', download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=4)

        valset = torchvision.datasets.CIFAR10(root='./data', train='val', download=True, transform=transform_test)
        valloader = torch.utils.data.DataLoader(valset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        testset = torchvision.datasets.CIFAR10(root='./data', train='test', download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=False, num_workers=4)


        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        nClasses = len(classes)

        return trainloader, testloader, valloader, nClasses

    #--------------------------------------------------------------------------------#
    def initializeModel(self, model_name, num_classes, use_pretrained=True):
        model_name = model_name.lower()
        model = None
        input_size = 0

        if "resnet" in model_name:
            """ Resnet18
            """
            if '18' in model_name:
                model = torchvision.models.resnet18(pretrained=use_pretrained)
            elif '34' in model_name:
                model = torchvision.models.resnet34(pretrained=use_pretrained)
            elif '50' in model_name:
                model = torchvision.models.resnet50(pretrained=use_pretrained)
            elif '101' in model_name:
                model = torchvision.models.resnet101(pretrained=use_pretrained)
            elif '152' in model_name:
                model = torchvision.models.resnet152(pretrained=use_pretrained)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "alexnet":
            """ Alexnet
            """
            model = torchvision.models.alexnet(pretrained=use_pretrained)
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs,num_classes)
            input_size = 224

        elif "vgg" in model_name:
            """ VGG
            """
            if '11' in model_name:
                if 'bn' in model_name:
                    model = torchvision.models.vgg11_bn(pretrained=use_pretrained)
                else:
                    model = torchvision.models.vgg11(pretrained=use_pretrained)
            elif '13' in model_name:
                if 'bn' in model_name:
                    model = torchvision.models.vgg13_bn(pretrained=use_pretrained)
                else:
                    model = torchvision.models.vgg13(pretrained=use_pretrained)
            elif '16' in model_name: 
                if 'bn' in model_name:
                    model = torchvision.models.vgg16_bn(pretrained=use_pretrained)
                else:
                    model = torchvision.models.vgg16(pretrained=use_pretrained)
            elif '19' in model_name:
                if 'bn' in model_name:
                    model = torchvision.models.vgg19_bn(pretrained=use_pretrained)
                else:
                    model = torchvision.models.vgg19(pretrained=use_pretrained)
            else:
                print("Invalid model name, returning 'None'...")
                return None

            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs,num_classes)
            input_size = 224

        elif "squeezenet" in model_name:
            """ Squeezenet
            """
            if '1_1' in model_name:
                model = torchvision.models.squeezenet1_1(pretrained=use_pretrained)
            else:
                model = torchvision.models.squeezenet1_0(pretrained=use_pretrained)
            num_ftrs = model.classifier[1].in_channels
            model.classifier[1] = nn.Conv2d(num_ftrs, num_classes, kernel_size=(1,1), stride=(1,1))
            model.num_classes = num_classes
            input_size = 224

        elif "densenet" in model_name:
            """ Densenet
            """
            if '161' in model_name:
                model = torchvision.models.densenet161(pretrained=use_pretrained)
            if '169' in model_name:
                model = torchvision.models.densenet169(pretrained=use_pretrained)
            if '201' in model_name:
                model = torchvision.models.densenet201(pretrained=use_pretrained)
            else: # else, get densenet121
                model = torchvision.models.densenet121(pretrained=use_pretrained)

            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "inception":
            """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            model = torchvision.models.inception_v3(pretrained=use_pretrained)
            # Handle the auxilary net
            num_ftrs = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs,num_classes)
            input_size = 299

        else:
            print("Invalid model name, returning 'None'...")
            return None

        return model, input_size
