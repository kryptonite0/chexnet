import os
import numpy as np
import time
import sys
from PIL import Image
import re

import pydicom

import cv2

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from DensenetModels import DenseNet121
from DensenetModels import DenseNet169
from DensenetModels import DenseNet201

from DatasetGenerator import dcm_to_png

#-------------------------------------------------------------------------------- 
#---- Class to generate heatmaps (CAM)

class HeatmapGenerator ():
    
    #---- Initialize heatmap generator
    #---- pathModel - path to the trained densenet model
    #---- nnArchitecture - architecture name DENSE-NET121, DENSE-NET169, DENSE-NET201
    #---- nnClassCount - class count, 14 for chxray-14

 
    def __init__ (self, pathModel, nnArchitecture, nnClassCount, transCrop):
       
        #---- Initialize the network
        if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(nnClassCount, True).cuda()
        elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, True).cuda()
        elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, True).cuda()
          
        model = torch.nn.DataParallel(model).cuda()

        print("=> loading checkpoint")
        modelCheckpoint = torch.load(pathModel)
        # https://github.com/KaiyangZhou/deep-person-reid/issues/23
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = modelCheckpoint['state_dict']
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
        print("=> loaded checkpoint")

        self.model = model.module.densenet121.features
        self.model.eval()
        
        #---- Initialize the weights
        self.weights = list(self.model.parameters())[-2]

        #---- Initialize the image transform - resize + normalize
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        transformList.append(transforms.Resize(transCrop))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)      
        
        self.transformSequence = transforms.Compose(transformList)
    
    #--------------------------------------------------------------------------------
     
    def generate (self, pathImageFile, pathOutputFile, transCrop):
        
        #---- Load image, transform, convert 
        if pathImageFile[-4:]=='.dcm':
            dcmimg = pydicom.dcmread(pathImageFile)
            imageData = dcm_to_png(dcmimg).convert('RGB')
        else:
            imageData = Image.open(pathImageFile).convert('RGB')

        imageData = self.transformSequence(imageData)
        imageData = imageData.unsqueeze_(0)
        
        input = torch.autograd.Variable(imageData)
        
        self.model.cuda()
        output = self.model(input.cuda())
        
        #---- Generate heatmap
        heatmap = None
        for i in range (0, len(self.weights)):
            map = output[0,i,:,:]
            if i == 0: heatmap = self.weights[i] * map
            else: heatmap += self.weights[i] * map
        
        #---- Blend original and heatmap 
        npHeatmap = heatmap.cpu().data.numpy()

        if pathImageFile[-4:]=='.dcm':
            imgOriginal = pydicom.dcmread(pathImageFile).pixel_array
        else:    
            imgOriginal = cv2.imread(pathImageFile, 1)
        imgOriginal = cv2.resize(imgOriginal, (transCrop, transCrop))
        
        cam = npHeatmap / np.max(npHeatmap)
        cam = cv2.resize(cam, (transCrop, transCrop))
#         heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
              
#         img = heatmap #* 0.5 + imgOriginal
            
#         cv2.imwrite(pathOutputFile, img)
        
        return cam
        
#-------------------------------------------------------------------------------- 

# pathInputImage = 'test/00009285_000.png'
# pathOutputImage = 'test/heatmap.png'
# pathModel = 'models/m-25012018-123527.pth.tar'

# nnArchitecture = 'DENSE-NET-121'
# nnClassCount = 14

# transCrop = 224

# h = HeatmapGenerator(pathModel, nnArchitecture, nnClassCount, transCrop)
# h.generate(pathInputImage, pathOutputImage, transCrop)
