#coding= utf-8
import os
import torch
from torchvision import transforms
from data_pipe import get_data 
from vgg import VGG_13
from resnet18 import ResNet18
import numpy as np
import cv2
from PIL import Image


class Infer(object):

    def __init__(self):
        self.model = ResNet18()
        self.model.load_state_dict(torch.load("./models/model_65.pth"))
        self.model.eval()
        self.cls = {' 0': 0, ' 1': 1, ' 10': 2, ' 11': 3, ' 12': 4, ' 13': 5, ' 14': 6, ' 15': 7, ' 16': 8, ' 17': 9, ' 18': 10, ' 19': 11, ' 2': 12, ' 20': 13, ' 21': 14, ' 22': 15, ' 23': 16, ' 24': 17, ' 25': 18, ' 26': 19, ' 27': 20, ' 28': 21, ' 29': 22, ' 3': 23, ' 30': 24, ' 31': 25, ' 32': 26, ' 33': 27, ' 34': 28, ' 35': 29, ' 36': 30, ' 37': 31, ' 38': 32, ' 39': 33, ' 4': 34, ' 5': 35, ' 6': 36, ' 7': 37, ' 8': 38, ' 9': 39}
        self.new_cls = dict(zip(self.cls.values(), self.cls.keys()))         

    def _infer(self, img_tensor):
        with torch.no_grad():
            result = self.model(img_tensor)
        return result

    def predict(self, path):
        img_path_list = [os.path.join(path ,x) for x in os.listdir(path)]
        transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        for img_path in img_path_list:
            print(img_path)
            img = cv2.imread(img_path)
            #img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img_tensor = transform(img)
            print("img_tensor shape =", img_tensor.size())
            #img_tensor = torch.from_numpy(np.asarray(img)).permute(2,0,1).float()/255.0
            img_tensor = img_tensor.reshape((1, 3, 224, 224))
            # _, preds = torch.max(outputs.data, dim = 1)
            result = self._infer(img_tensor)
            _, preds = torch.max(result.data, dim = 1)
            print(self.new_cls[preds.numpy()[0]].strip())
            

if __name__ == "__main__":
    path = "./test_images"
    Infer().predict(path)



