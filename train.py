#coding= utf-8
import os
import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
import pandas as pd
import datetime
from sklearn.metrics import roc_auc_score
from vgg import VGG_13
from model import SimpleNet
from resnet18 import ResNet18
from data_pipe import get_data 

# basic params
path = "./data/garbageClassifyData"
epochs = 80
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# single GPU
#model = SimpleNet()
# multi GPU
# model = nn.DataParallel(models.resnext50_32x4d(num_classes=40))
model = nn.DataParallel(ResNet18())
model.to(device)
model.optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
model.loss_func = torch.nn.CrossEntropyLoss()
#model.scheduler = torch.optim.lr_scheduler.StepLR(model.optimizer, step_size=10, gamma=0.1)\
model.scheduler = torch.optim.lr_scheduler.MultiStepLR(model.optimizer, milestones=[30, 50, 65, 75], gamma=0.1)

#model.metric_func = lambda y_pred,y_true: roc_auc_score(y_true.data.numpy(),y_pred.data.numpy())
dl_train, dl_val = get_data(path)

def train_v1(model, epochs, dl_train, dl_val):
    print("Strat Training......")
    for epoch in range(1, epochs+1):
        #model.scheduler.step()
        # Train
        running_loss = 0.0
        running_metric = 0.0
        running_corrects = 0
        for i, data in enumerate(dl_train, 0):
            # train model, dropout
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)  
            model.train()
            model.optimizer.zero_grad()
            outputs = model(inputs)
            labels = labels.squeeze()
            # calc acc
            _, preds = torch.max(outputs.data, dim = 1)
            running_corrects += torch.sum(preds == labels.data)            

            loss = model.loss_func(outputs, labels.long())
            #metric = model.metric_func(outputs, labels)
            loss.backward()
            model.optimizer.step()
            running_loss += loss.item()
            #running_metric += metric.item()
            #_, predicted = torch.max(outputs.data, dim = 1)
            
            if (i%50) == 49:
                #print("[Train]: ", epoch, i+1, running_loss/100, "Auc=", running_metric/100)
                print("[Train]: ", epoch, i+1, running_loss/50, model.optimizer.state_dict()['param_groups'][0]['lr'])
                print("Acc =", float(running_corrects)/2400)
                running_loss = 0.0
                running_metric = 0.0
                running_corrects = 0
        model.scheduler.step()
        """
        # Eval
        val_loss = 0.0
        val_metric = 0.0
        val_step = 0
        for step, data_val in enumerate(dl_val, 0):
            inputs_val, labels_val = data_val
            inputs_val = inputs_val.to(device)
            labels_val = labels_val.to(device)
            model.eval()
            with torch.no_grad():
                outputs_val = model(inputs_val)
                loss_val = model.loss_func(outputs_val, labels_val)
                metric_val = model.metric_func(outputs_val.cpu(), labels_val.cpu())
                val_loss += loss_val.item()
                val_metric += metric_val.item()
            val_step = step
        print("Eval : epochs={} step={} loss_val={} Auc={}".format(epoch, val_step, val_loss/val_step, val_metric/val_step))
        # save_model
        """
        if (epoch%5) == 0:
            model_name = "model_" + str(epoch) + ".pth"
            torch.save(model.module.state_dict(), os.path.join("./models", model_name))
        
train_v1(model, epochs, dl_train, dl_val)

