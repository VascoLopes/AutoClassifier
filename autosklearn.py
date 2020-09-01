import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
import pretrainedmodels
import numpy as np
from torchvision import transforms, datasets
from split_train import get_train_valid_loader
from sklearn import metrics
from autoPyTorch import AutoNetClassification
# data and metric imports
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import json

import h2o
from h2o.automl import H2OAutoML

import utils
from split_train import get_train_valid_loader
import csv
import os
import time

path         = "./dataset/public/"
path_models  = "/dataset/models/"
batch_size   = 10
device       = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
utils        = utils.Utils(batch_size, device)

network = 'vgg16'
model = utils.initializeModel(network, 2)


# Generate dataset features, removing VGG16 classification component
def generateDatasetFeatures(network):
    for problem in range(1,7):
        trainLoader, valLoader, testLoader = get_train_valid_loader(batch_size, 0, path+"Problem"+str(problem)+"/", problem, 0.3, num_workers=4)
        model = torch.load(path_models+network+"_"+"problem"+str(problem)+".pth")
        model.classifier[1] = nn.Identity()
        model.classifier[2] = nn.Identity()
        model.classifier[3] = nn.Identity()
        model.classifier[4] = nn.Identity()
        model.classifier[5] = nn.Identity()
        model.classifier[6] = nn.Identity()

        print("Test")
        f = open('./dataset/'+network+'_linear_problem'+str(problem)+'_test.csv','w')
        #f.write('x\ty\n')

        inferenceTime = []
        with torch.no_grad():
            model.eval()
            for data in testLoader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                #stBatch = time.time()
                outputs = model(inputs)
                #etBatch = time.time()-stBatch
                #inferenceTime.append(etBatch)
                for output, label in zip(outputs.cpu().numpy(), labels.cpu().numpy()):
                    #print(np.array2string(output)+"\t"+np.array2string(label))
                    for value in output:
                        f.write(str(value)+",")
                    f.write(str(label))
                    #f.write(str(output.tolist())+","+str(label)) #Give your csv text here.
                    f.write("\n")
        f.close()
        #inferenceTimeMean = [i/batch_size for i in inferenceTime] # list batches time
        #print (inferenceTimeMean)
        #print(f'Inference Time Mean: {np.mean(inferenceTimeMean)}, STD:{np.std(inferenceTimeMean)}')
        
        print("Validation")
        f = open('./dataset/'+network+'_linear_problem'+str(problem)+'_validation.csv','w')
        with torch.no_grad():
            for data in valLoader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                for output, label in zip(outputs.cpu().numpy(), labels.cpu().numpy()):
                    for value in output:
                        f.write(str(value)+",")
                    f.write(str(label))
                    f.write("\n")
        f.close()

        print("Train")
        f = open('./dataset/'+network+'_linear_problem'+str(problem)+'_train.csv','w')
        with torch.no_grad():
            for data in trainLoader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                for output, label in zip(outputs.cpu().numpy(), labels.cpu().numpy()):
                    for value in output:
                        f.write(str(value)+",")
                    f.write(str(label))
                    f.write("\n")
        f.close()



if __name__ == "__main__":
    h2o.init()

    for problem in range(1,7):
        print("[Starting Problem "+str(problem)+" ...]")
        # put path for the newly datasets generated before
        network= 'vgg16'
        x_train = h2o.import_file('./dataset/'+network+'_linear_problem'+str(problem)+'_train.csv')
        x_val = h2o.import_file('./dataset/'+network+'_linear_problem'+str(problem)+'_validation.csv')
        x_test = h2o.import_file('./dataset/'+network+'_linear_problem'+str(problem)+'_test.csv')
        y_test = x_test['C4097'] #predictions
        x = x_train.columns
        y = 'C4097'
        x.remove(y)
        #x_train[y] = x_train[y].asfactor()
        #x_val[y] = x_val[y].asfactor()
        #x_test[y] = x_test[y].asfactor()

        
        aml = H2OAutoML(max_models = 30, max_runtime_secs=int(3600*2), seed = 1) #each problem will be searched for 2 hours
        aml.train(y = y, training_frame = x_train, validation_frame=x_val)

        lb = aml.leaderboard
        print(lb.head())

        startTime = time.time()
        preds = aml.predict(x_test)
        print("Predictions")
        endTime = time.time()-startTime
        print (f'Prediction time: {endTime} secs')
        print (f'Prediction time / individual: {endTime/173} secs')
        print(preds)
        print()
        lb = h2o.automl.get_leaderboard(aml, extra_columns = 'ALL')
        print(lb)
        #h2o.save_model(aml.leader, path = "./AutoML_models/problem"+str(problem)+"/")
        
        true_label = np.rint(np.array(h2o.as_list(x_test[y]))).astype(int)
        predictions = np.rint(np.array(h2o.as_list(preds))).astype(int)
        print("Metrics [...]")
        fpr, tpr, thresholds = metrics.roc_curve(true_label, predictions)
        auc        = metrics.auc(fpr, tpr)
        accuracy   = sklearn.metrics.accuracy_score(true_label, predictions)
        confMatrix = metrics.confusion_matrix(true_label, predictions)
        print(f'Accuracy:{accuracy}')
        print(f'AUC-Score:{auc}')
        print(f'Confusion Matrix:\n{confMatrix}')