import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
import pretrainedmodels
import numpy as np
from torchvision import transforms, datasets
from split_train import get_train_valid_loader
from sklearn import metrics
from autosklearn import generateDatasetFeatures

from utils import Utils
import time
import copy
import os
import random

path         = "./dataset/public/"
problems     = 6 # Class1,Class2,Class3,Class4,Class5,Class6
classes      = 2 # 0-Defect; 1-No Defect
input_shape  = 512 # 512*512
batch_size   = 10
device       = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
epochs       = 100
utils        = Utils(batch_size, device)

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":  
    # Iterate over all problems
    networks = ['vgg16']#['vgg11', 'vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'densenet121']
    
    for network in networks:
        for problem in range(1,7):
            seed_torch(0)
            print("[Starting Problem "+str(problem)+" ...]")
            
            # Create model
            print("[Creating the model ...]")
            print(f"{network}")

            # Model selection
            model, _ = utils.initializeModel(network, 2, True)
            model.to(device)

            trainLoader, valLoader, testLoader = get_train_valid_loader(batch_size, 0, path+"Problem"+str(problem)+"/", problem, 0.3)

            ## Weight dataset loss
            #weights = torch.tensor([1., 10.])
            criterion = utils.getCrossEntropyLoss()
            optimizer_conv = utils.getSGDOptimizer(model)

            regular_train_acc, regular_train_loss = [], []
            regular_val_acc, regular_val_loss = [], []
            # Store model in best val acc, best val loss
            best_model_wts = copy.deepcopy(model.state_dict())
            best_model_lowest_loss = copy.deepcopy(model.state_dict())
            best_acc_val = [0.0, 0] # accuracy, epoch
            best_loss_val = [20.0, 0] # loss epoch

            # Time for training
            startTimeTrain = time.time()

            print("### Start Training [...] ###")
            for epoch in range(0,epochs):
                # Train
                model, train_acc, train_loss, predictions = utils.train(model, trainLoader, criterion, optimizer_conv, epoch)
                regular_train_acc.append(train_acc)
                regular_train_loss.append(train_loss)

                # Validation
                val_acc, val_loss, predictions = utils.evaluate(model, valLoader, criterion, epoch)
                regular_val_acc.append(val_acc)
                regular_val_loss.append(val_loss)

                if val_acc > best_acc_val[0]: # store best model so far, for later, based on best val acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_acc_val[0], best_acc_val[1] = val_acc, epoch
                if val_loss < best_loss_val[0]: # store best model according to loss
                    best_model_lowest_loss = copy.deepcopy(model.state_dict())
                    best_loss_val[0], best_loss_val[1] = val_loss, epoch

            endTimeTrain = time.time()

            model.load_state_dict(best_model_wts)
            accuracyTest, predictions, trueLabels, inferenceTime = utils.test(model, testLoader)

            fpr, tpr, thresholds = metrics.roc_curve(trueLabels, predictions)
            auc        = metrics.auc(fpr, tpr)
            confMatrix = metrics.confusion_matrix(trueLabels, predictions)
            recall     = metrics.recall_score(trueLabels, predictions, labels=np.unique(predictions), pos_label=0)
            precision  = metrics.precision_score(trueLabels, predictions, labels=np.unique(predictions), pos_label=0)

            print ("### Test metrics ###")
            print(f'Best Val Accuracy:{best_acc_val[0]:0.4}')
            print(f'Accuracy: {accuracyTest:0.4}')
            print(f'AUC-ROC: {auc}')
            print(f'Confusion Matrix:\n{confMatrix}')
            print(f'Recall: {recall}')
            print(f'Precision: {precision}')

            print("### Time Metrics ###")
            print(f'Time to train: {endTimeTrain-startTimeTrain}')

            inferenceTimeMean = [i/batch_size for i in inferenceTime] # list batches time
            #print (inferenceTimeMean)
            print(f'Inference Time Mean: {np.mean(inferenceTimeMean)}, STD:{np.std(inferenceTimeMean)}')

            torch.save(model, "./"+modelName+"_problem"+str(problem)+".pth")