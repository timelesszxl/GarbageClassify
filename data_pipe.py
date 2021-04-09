#coding=utf-8
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

def get_data(path):
    transform_train = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_valid = transforms.Compose([
        transforms.Resize([224, 224]),
	transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    ds_train = datasets.ImageFolder(os.path.join(path, "train"), 
        transform = transform_train, target_transform= lambda t:torch.tensor([t]).float())
    ds_valid = datasets.ImageFolder(os.path.join(path, "test"),
        transform = transform_train, target_transform= lambda t:torch.tensor([t]).float())
    #"""
    #ds_train = datasets.ImageFolder(os.path.join(path, "train"), transform=transform_train)
    #ds_valid = datasets.ImageFolder(os.path.join(path, "test"), transform=transform_valid)
    dl_train = DataLoader(ds_train, batch_size=48, shuffle=True, num_workers=8)
    dl_valid = DataLoader(ds_valid, batch_size=48, shuffle=True, num_workers=8)
    # PyTorch img = [B, C, H, W]
    #for x,y in dl_train:
    #    print(x.shape, y)
    print(ds_train.class_to_idx)
    print(ds_valid.class_to_idx)
    return dl_train, dl_valid

if __name__ == "__main__":
    get_data("./data/garbageClassifyData")
