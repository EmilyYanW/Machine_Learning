
# coding: utf-8

# In[1]:


from __future__ import division
import os, sys, time, pickle
import itertools

get_ipython().run_line_magic('matplotlib', 'inline')
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (8.0, 10.0)

import cv2
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('jpg')
from PIL import Image, ImageDraw

import torch
import torch.nn as nn

from math import ceil, floor 


# ### Load Data

# In[17]:


dataType = 'train2014'
dataDir='' 
annFile_train='instances_{}.json'.format(dataType)
coco_train=COCO(annFile_train)

cats = coco_train.loadCats(coco_train.getCatIds()) # categories
cat_id_to_name = {cat['id']: cat['name'] for cat in cats} # category id to name mapping
cat_name_to_id = {cat['name']: cat['id'] for cat in cats} # category name to id mapping

cat_to_supercat = {cat['name']: cat['supercategory'] for cat in cats}
cat_id_to_supercat = {cat['id']: cat['supercategory'] for cat in cats}


# In[18]:


## Validation
dataType = 'val2014'
dataDir='' 
annFile_val='instances_{}.json'.format(dataType)
coco_val=COCO(annFile_val)


# In[19]:


## Test
dataType = 'test2014'
dataDir='' 
annFile_test='instances_{}.json'.format(dataType)
coco_test=COCO(annFile_test)


# In[26]:


dataType = 'train2014_2000_small'
[img_list_train, feats_train] = pickle.load(open(os.path.join(dataDir, 
                                              '{}.p'.format(dataType)),
                                 'rb'),encoding='latin1')


# In[27]:


dataType = 'val2014_2000_small'
[img_list_val, feats_val] = pickle.load(open(os.path.join(dataDir, 
                                              '{}.p'.format(dataType)),
                                 'rb'),encoding='latin1')


# In[28]:


dataType = 'test2014_2000_small'
[img_list_test, feats_test] = pickle.load(open(os.path.join(dataDir, 
                                              '{}.p'.format(dataType)),
                                 'rb'),encoding='latin1')


# In[29]:


C = [key for key, value in cat_id_to_supercat.items() 
              if value =='vehicle' or value =='animal' ]
C_id = {j:i for (i,j) in enumerate (C)}


# In[30]:


C


# In[31]:


# IoU
def iou(rect1, rect2): # rect = [x, y, w, h]
    x1, y1, w1, h1 = rect1
    X1, Y1 = x1+w1, y1 + h1
    x2, y2, w2, h2 = rect2
    X2, Y2 = x2+w2, y2 + h2
    a1 = (X1 - x1 + 1) * (Y1 - y1 + 1)
    a2 = (X2 - x2 + 1) * (Y2 - y2 + 1)
    x_int = max(x1, x2) 
    X_int = min(X1, X2) 
    y_int = max(y1, y2) 
    Y_int = min(Y1, Y2) 
    a_int = (X_int - x_int + 1) * (Y_int - y_int + 1) * 1.0 
    if x_int > X_int or y_int > Y_int:
        a_int = 0.0 
    return a_int / (a1 + a2 - a_int)  


# In[830]:


boxes_train = np.array(pickle.load(open("train2014_bboxes.p", "rb"),encoding='latin1'))
boxes_val  = np.array(pickle.load(open("val2014_bboxes.p", "rb"),encoding='latin1'))
boxes_test  = np.array(pickle.load(open("test2014_bboxes.p", "rb"),encoding='latin1'))


# In[831]:


boxes_train


# In[832]:


boxes_val


# In[833]:


boxes_test


# In[834]:


import pandas as pd
box_cat = []
for i,id in enumerate(img_list_train):
    if i%100 == 0:
        print(i)
        
    img = coco_train.loadImgs([id])[0]
    annIds = coco_train.getAnnIds(imgIds=img['id'],  iscrowd=None)
    anns = coco_train.loadAnns(annIds)

    
    ## Get true boxes
    true_bboxes = []
    true_bboxes_cat = []
    
    for ann in anns:
        x, y, w, h = ann['bbox']
        true_bboxes += [ann['bbox']]
        true_bboxes_cat.append(ann['category_id'])

    bboxes = boxes_train[1][i]
    bboxes_cat = []
    
    if bboxes is not None:
        
        for box in bboxes:
            y = np.zeros(23)
            y[18] = id
            y[19] = box[0]
            y[20] = box[1]
            y[21] = box[2]
            y[22] = box[3]
            
            for z,true_box in enumerate(true_bboxes):
                score = iou(true_box, box)
                if score > 0.5:
                    category = true_bboxes_cat[z]
                    y[C_id[category]] = 1
            box_cat.append(y)


# In[835]:


box_cat


# In[839]:


import pandas as pd

box_cat_val = []

for i,id in enumerate(img_list_val):
    if i%100 == 0:
        print(i)
    img = coco_val.loadImgs([id])[0]
    annIds = coco_val.getAnnIds(imgIds=img['id'],  iscrowd=None)
    anns = coco_val.loadAnns(annIds)

    
    ## Get true boxes
    true_bboxes = []
    true_bboxes_cat = []
    
    for ann in anns:
        x, y, w, h = ann['bbox']
        true_bboxes += [ann['bbox']]
        true_bboxes_cat.append(ann['category_id'])

    bboxes = boxes_val[1][i]
    bboxes_cat = []
    
    if bboxes is not None:
        
        for box in bboxes:
            y = np.zeros(23)
            y[18] = id
            y[19] = box[0]
            y[20] = box[1]
            y[21] = box[2]
            y[22] = box[3]
            
            for z,true_box in enumerate(true_bboxes):
                score = iou(true_box, box)
                if score > 0.5:
                    category = true_bboxes_cat[z]
                    y[C_id[category]] = 1
            box_cat_val.append(y)


# In[840]:


box_cat_val


# In[841]:


import pandas as pd

box_cat_test = []

for i,id in enumerate(img_list_test):
    if i%100 == 0:
        print(i)
    img = coco_test.loadImgs([id])[0]
    annIds = coco_test.getAnnIds(imgIds=img['id'],  iscrowd=None)
    anns = coco_test.loadAnns(annIds)

    
    ## Get true boxes
    true_bboxes = []
    true_bboxes_cat = []
    
    for ann in anns:
        x, y, w, h = ann['bbox']
        true_bboxes += [ann['bbox']]
        true_bboxes_cat.append(ann['category_id'])

    bboxes = boxes_test[1][i]
    bboxes_cat = []
    
    if bboxes is not None:
        
        for box in bboxes:
            y = np.zeros(23)
            y[18] = id
            y[19] = box[0]
            y[20] = box[1]
            y[21] = box[2]
            y[22] = box[3]
            
            for z,true_box in enumerate(true_bboxes):
                score = iou(true_box, box)
                if score > 0.5:
                    category = true_bboxes_cat[z]
                    y[C_id[category]] = 1
            box_cat_test.append(y)


# In[842]:


box_cat_array = np.array(box_cat)
box_cat_array_val = np.array(box_cat_val)
box_cat_array_test = np.array(box_cat_test)


# In[843]:


pickle.dump(box_cat_array, open( "box_train.p", "wb" ) )
pickle.dump(box_cat_array_val, open( "box_val.p", "wb" ) )
pickle.dump(box_cat_array_test, open( "box_test.p", "wb" ) )


# In[838]:


box_cat_array[:,18:]


# In[2]:


box_cat_array = np.array(pickle.load(open("box_train.p", "rb"),encoding='latin1'))
box_cat_array_val = np.array(pickle.load(open("box_val.p", "rb"),encoding='latin1'))
box_cat_array_test = np.array(pickle.load(open("box_test.p", "rb"),encoding='latin1'))


# In[3]:


box_cat_array


# In[4]:


box_cat_array_val


# In[5]:


box_cat_array_test


# ### Project boxes onto images and get features

# In[6]:


# nearest neighbor in 1-based indexing
def _nnb_1(x):                                                                                                                               
    x1 = int(floor((x + 8) / 16.0))
    x1 = max(1, min(x1, 13))
    return x1


def project_onto_feature_space(rect, image_dims):
    # project bounding box onto conv net
    # @param rect: (x, y, w, h)
    # @param image_dims: (imgx, imgy), the size of the image
    # output bbox: (x, y, x'+1, y'+1) where the box is x:x', y:y'

    # For conv 5, center of receptive field of i is i_0 = 16 i for 1-based indexing
    imgx, imgy = image_dims
    x, y, w, h = rect
    # scale to 224 x 224, standard input size.
    x1, y1 = ceil((x + w) * 224 / imgx), ceil((y + h) * 224 / imgy)
    x, y = floor(x * 224 / imgx), floor(y * 224 / imgy)
    px = _nnb_1(x + 1) - 1 # inclusive
    py = _nnb_1(y + 1) - 1 # inclusive
    px1 = _nnb_1(x1 + 1) # exclusive
    py1 = _nnb_1(y1 + 1) # exclusive

    return [px, py, px1, py1]


# In[7]:


class Featurizer:
    dim = 11776 # for small features
    def __init__(self):
        # pyramidal pooling of sizes 1, 3, 6
        self.pool1 = nn.AdaptiveMaxPool2d(1)
        self.pool3 = nn.AdaptiveMaxPool2d(3)                                                                                                 
        self.pool6 = nn.AdaptiveMaxPool2d(6)
        self.lst = [self.pool1, self.pool3, self.pool6]
        
    def featurize(self, projected_bbox, image_features):
        # projected_bbox: bbox projected onto final layer
        # image_features: C x W x H tensor : output of conv net
        full_image_features = torch.from_numpy(image_features)
        x, y, x1, y1 = projected_bbox
        crop = full_image_features[:, x:x1, y:y1] 
#         return torch.cat([self.pool1(crop).view(-1), self.pool3(crop).view(-1),  
#                           self.pool6(crop).view(-1)], dim=0) # returns torch Variable
        return torch.cat([self.pool1(crop).view(-1), self.pool3(crop).view(-1),  
                          self.pool6(crop).view(-1)], dim=0).data.numpy() # returns numpy array


# In[ ]:


### For Category 2


# In[846]:


for c in C:
    print(str(c )+' '+ str(cat_id_to_name.get(c)))


# In[11]:


category = 1


# In[67]:


# Get postive boxes
Positive = box_cat_array[box_cat_array[:,category] == 1.]
Positive_val = box_cat_array_val[box_cat_array_val[:,category] == 1.]
Positive_test = box_cat_array_test[box_cat_array_test[:,category] == 1.]


# In[68]:


print(len(box_cat_array))
print(len(Positive))

print(len(box_cat_array_val))
print(len(Positive_val))

print(len(box_cat_array_test))
print(len(Positive_test))


# In[14]:


### Randomly Find 3000 boxes
R = 2000
if R > len(Positive):
    R = len(Positive)
    
r = np.random.choice(range(0,len(Positive)), size=R, replace=False)

Positive = Positive[r]


# In[ ]:


r = np.random.choice(range(0,len(Positive)), size=5000, replace=False)

Positive = Positive[r]


# In[15]:


print(len(box_cat_array))
print(len(Positive))

print(len(box_cat_array_val))
print(len(Positive_val))

print(len(box_cat_array_test))
print(len(Positive_test))


# In[22]:


Image_Id = Positive[7][18]
img = coco_train.loadImgs([Image_Id])[0]
img_pil = Image.open('%s/%s'%('/home/ubuntu/images/train',img['file_name']))
img_pil


# In[23]:


Image_Id = Positive_val[1][18]
img = coco_val.loadImgs([Image_Id])[0]
img_pil = Image.open('%s/%s'%('/home/ubuntu/images/val',img['file_name']))
img_pil


# In[24]:


Image_Id = Positive_test[1][18]
img = coco_test.loadImgs([Image_Id])[0]
img_pil = Image.open('%s/%s'%('/home/ubuntu/images/test',img['file_name']))
img_pil


# In[32]:


# Get features for boxes that are positive for Category 
Positive_feats = []

for posi in Positive:
    
    Image_Id = posi[18]
    img = coco_train.loadImgs([Image_Id])[0]
    
    img_pil = Image.open('%s/%s'%('/home/ubuntu/images/train',img['file_name']))
    
    box = posi[19:23]
    projected_bbox = project_onto_feature_space(box, img_pil.size)
    
    idx = img_list_train.index(Image_Id)
    
    img_feats = feats_train[idx]
    
    featurizer = Featurizer()
    bbox_feats = featurizer.featurize(projected_bbox, img_feats) 
    
    Positive_feats.append(bbox_feats)


# In[33]:


# Get features for boxes that are positive for Category 
# Validation

Positive_feats_val = []

for posi in Positive_val:
    
    Image_Id = posi[18]
    img = coco_val.loadImgs([Image_Id])[0]
    
    img_pil = Image.open('%s/%s'%('/home/ubuntu/images/val',img['file_name']))
    
    box = posi[19:23]
    projected_bbox = project_onto_feature_space(box, img_pil.size)
    
    idx = img_list_val.index(Image_Id)
    
    img_feats = feats_val[idx]
    
    featurizer = Featurizer()
    bbox_feats = featurizer.featurize(projected_bbox, img_feats) 
    
    Positive_feats_val.append(bbox_feats)


# In[34]:


# Get features for boxes that are positive for Category 
# Test

Positive_feats_test = []

for posi in Positive_test:
    
    Image_Id = posi[18]
    img = coco_test.loadImgs([Image_Id])[0]
    
    img_pil = Image.open('%s/%s'%('/home/ubuntu/images/test',img['file_name']))
    
    box = posi[19:23]
    projected_bbox = project_onto_feature_space(box, img_pil.size)
    
    idx = img_list_test.index(Image_Id)
    
    img_feats = feats_test[idx]
    
    featurizer = Featurizer()
    bbox_feats = featurizer.featurize(projected_bbox, img_feats) 
    
    Positive_feats_test.append(bbox_feats)


# In[35]:


Positive_feat = np.array(Positive_feats)
Positive_feat_val = np.array(Positive_feats_val)
Positive_feat_test = np.array(Positive_feats_test)


# In[36]:


# Randomly get 2*n of features for boxes that are negative 
Negative = box_cat_array[box_cat_array[:,category] != 1.]

N_train = len(Positive)

# R = 3000*10
# if R > len(Negative):
#     R = len(Negative)
    
# r = np.random.choice(range(0,len(Negative)), size=R, replace=False)

# Negative = Negative[r]

j_train = np.random.choice(range(0,len(Negative)), size=2*N_train, replace=False)


# In[37]:


# Randomly get 2*n of features for boxes that are negative 
# Validation

Negative_val = box_cat_array_val[box_cat_array_val[:,category] != 1.]

N_val = len(Positive_val)

j_val = np.random.choice(range(0,len(Negative_val)), size=2*N_val, replace=False)


# In[38]:


# Randomly get 2*n of features for boxes that are negative 
# Test

Negative_test = box_cat_array_test[box_cat_array_test[:,category] != 1.]

N_test = len(Positive_test)

j_test = np.random.choice(range(0,len(Negative_test)), size=2*N_test, replace=False)


# In[39]:


Negative_feats = []

for nega in Negative[j_train]:
    
    Image_Id = nega[18]
    
    img = coco_train.loadImgs([Image_Id])[0]
    
    img_pil = Image.open('%s/%s'%('/home/ubuntu/images/train',img['file_name']))
    
    box = nega[19:23]
    projected_bbox = project_onto_feature_space(box, img_pil.size)
    
    idx = img_list_train.index(Image_Id)
    
    img_feats = feats_train[idx]
    
    featurizer = Featurizer()
    bbox_feats = featurizer.featurize(projected_bbox, img_feats) 
    
    Negative_feats.append(bbox_feats)


# In[40]:


# Validation
Negative_feats_val = []

for nega in Negative_val[j_val]:
    
    Image_Id = nega[18]
    
    img = coco_val.loadImgs([Image_Id])[0]
    
    img_pil = Image.open('%s/%s'%('/home/ubuntu/images/val',img['file_name']))
    
    box = nega[19:23]
    projected_bbox = project_onto_feature_space(box, img_pil.size)
    
    idx = img_list_val.index(Image_Id)
    
    img_feats = feats_val[idx]
    
    featurizer = Featurizer()
    bbox_feats = featurizer.featurize(projected_bbox, img_feats) 
    
    Negative_feats_val.append(bbox_feats)


# In[41]:


# Test

Negative_feats_test = []

for nega in Negative_test[j_test]:
    
    Image_Id = nega[18]
    
    img = coco_test.loadImgs([Image_Id])[0]
    
    img_pil = Image.open('%s/%s'%('/home/ubuntu/images/test',img['file_name']))
    
    box = nega[19:23]
    projected_bbox = project_onto_feature_space(box, img_pil.size)
    
    idx = img_list_test.index(Image_Id)
    
    img_feats = feats_test[idx]
    
    featurizer = Featurizer()
    bbox_feats = featurizer.featurize(projected_bbox, img_feats) 
    
    Negative_feats_test.append(bbox_feats)


# In[42]:


Negative_feat = np.array(Negative_feats)
Negative_feat_val = np.array(Negative_feats_val)
Negative_feat_test = np.array(Negative_feats_test)


# In[43]:


# Combine the dataset
feat_train = np.concatenate((Positive_feat,Negative_feat), axis = 0)


# In[44]:


feat_val = np.concatenate((Positive_feat_val,Negative_feat_val), axis = 0)


# In[45]:


feat_test = np.concatenate((Positive_feat_test,Negative_feat_test), axis = 0)


# In[46]:


# Get the y array
y_train = np.concatenate((np.ones(N_train), np.zeros(N_train*2)), axis = 0)
y_val = np.concatenate((np.ones(N_val), np.zeros(N_val*2)), axis = 0)
y_test = np.concatenate((np.ones(N_test), np.zeros(N_test*2)), axis = 0)


# In[47]:


import torch
import random
from torch.autograd import Variable
from sklearn.metrics import average_precision_score


# In[49]:


dtype = torch.DoubleTensor 
w1 = Variable(torch.nn.init.normal_(torch.Tensor(11776,1).type(dtype))/1000, 
              requires_grad = True) 

losses_train_linear = []
AP_train_linear = []

losses_val_linear = []
AP_val_linear = []

z1=0
beta = 0.9
lambda1 = 1000


# In[51]:


# Logistic Regression


r = 1e-8
 
NN = len(y_train)
n = 100
NN_val = len(y_val)

for t in range(500*50):

    i = np.random.choice(range(0,NN), size=n, replace=False)

    x = Variable(torch.from_numpy(feat_train[i]).type(dtype), 
                      requires_grad=False)   

    y_pred = (x.view(n,11776)).mm(w1)

    y_original = Variable(torch.from_numpy(y_train[i]).type(dtype), 
                          requires_grad=False)

    loss = lambda1/2*w1.pow(2).sum() +            (y_original.view(n,1)*torch.log(1+torch.exp(-y_pred)) +                  (1-y_original.view(n,1))*torch.log(1+torch.exp(y_pred))).sum()/n
        
       # Print the training loss on random 2000 data points
    if t%500 ==0:
        print(t/500)
        S = 2000
        if S>NN_val:
            S = NN_val
        t = np.random.choice(range(0,NN), size=S, replace=False)
        x_2000 = Variable(torch.from_numpy(feat_train[t]).type(dtype), 
                      requires_grad=False)
        y_2000 = Variable(torch.from_numpy(y_train[t]).type(dtype), 
                          requires_grad=False)
               
        y_pred_2000 = (x_2000.view(S,11776)).mm(w1)
        gap_all = y_2000 - y_pred_2000
        
        loss_all = lambda1/2*w1.pow(2).sum()+            (y_2000.view(S,1)*torch.log(1+torch.exp(-y_pred_2000)) +                  (1-y_2000.view(S,1))*torch.log(1+torch.exp(y_pred_2000))).sum()/S
            
        losses_train_linear.append(loss_all.data[0])
        print(loss_all.data[0])
        
        y_pred_2000_pro = 1/(1+torch.exp(-y_pred_2000))
        
        precision = average_precision_score(np.array(y_2000.data), 
                                            np.array(y_pred_2000_pro.data),
                                            average='micro')
        AP_train_linear.append(precision)

        ## Validation

        t_val = np.random.choice(range(0,NN_val), size=S, replace=False)
        
        x_2000_val = Variable(torch.from_numpy(feat_val[t_val]).type(dtype), 
                      requires_grad=False)
        y_2000_val = Variable(torch.from_numpy(y_val[t_val]).type(dtype), 
                          requires_grad=False)
               
        y_pred_2000_val = (x_2000_val.view(S,11776)).mm(w1)
        
        gap_all_val = y_2000_val - y_pred_2000_val
        
        loss_all_val = lambda1/2*w1.pow(2).sum()+            (y_2000_val.view(S,1)*torch.log(1+torch.exp(-y_pred_2000_val)) +                  (1-y_2000_val.view(S,1))*torch.log(1+torch.exp(y_pred_2000_val))).sum()/S
            
        losses_val_linear.append(loss_all_val.data[0])
        
        print(loss_all_val.data[0])
        
        y_pred_2000_pro_val = 1/(1+torch.exp(-y_pred_2000_val))
        
        precision_val = average_precision_score(np.array(y_2000_val.data), 
                                            np.array(y_pred_2000_pro_val.data),
                                            average='micro')
        AP_val_linear.append(precision_val)

    loss.backward()
    
    z1= beta*z1+w1.grad.data
    w1.data -= r *z1

    w1.grad.data.zero_()


# In[52]:


AP_train_linear


# In[53]:


AP_val_linear


# In[54]:


plt.figure(figsize=(10,13))
plt.plot(AP_train_linear, label = "Train")
plt.plot(AP_val_linear, label = "Val")


# ### Update

# In[2298]:


X = Variable(torch.from_numpy(feat_train).type(dtype), 
                  requires_grad=False)   

Y_pred = (X.view(NN,11776)).mm(w1)

Y_original = Variable(torch.from_numpy(y_train).type(dtype), 
                      requires_grad=False)


## Get the N maximum negatives into feat_train_HN
Y_pred_Negative = Y_pred[-2*N_train:]



IDs = np.argpartition(Y_pred_Negative.detach().numpy().reshape(1,N_train*2)[0], -N_train)[-N_train:]
print(max(Y_pred_Negative[IDs].detach().numpy().reshape(1, N_train)[0]))
IDs = IDs + N_train

feat_train_HN = feat_train[IDs]
feat_train_HN = np.array(feat_train_HN)


## Randomly Sample N negatives into Negative_feats
j_train = np.random.choice(range(0,len(Negative)), size=N_train, replace=False)

Negative_feats = []

for nega in Negative[j_train]:

    Image_Id = nega[18]
    img = coco_train.loadImgs([Image_Id])[0]

    img_pil = Image.open('%s/%s'%('/home/ubuntu/images/train',img['file_name']))

    box = nega[19:23]
    projected_bbox = project_onto_feature_space(box, img_pil.size)

    idx = img_list_train.index(Image_Id)
    img_feats = feats_train[idx]

    featurizer = Featurizer()
    bbox_feats = featurizer.featurize(projected_bbox, img_feats) 

    Negative_feats.append(bbox_feats)
    
Negative_feat = np.array(Negative_feats)

feat_train = np.concatenate((Positive_feat,Negative_feat, feat_train_HN), axis = 0)   


# In[ ]:


## Randomly Sample N negatives into Negative_feats
j_train = np.random.choice(range(0,len(Negative)), size=N_train*2, replace=False)

Negative_feats = []

for nega in Negative[j_train]:

    Image_Id = nega[18]
    img = coco_train.loadImgs([Image_Id])[0]

    img_pil = Image.open('%s/%s'%('/home/ubuntu/images/train',img['file_name']))

    box = nega[19:23]
    projected_bbox = project_onto_feature_space(box, img_pil.size)

    idx = img_list_train.index(Image_Id)
    img_feats = feats_train[idx]

    featurizer = Featurizer()
    bbox_feats = featurizer.featurize(projected_bbox, img_feats) 

    Negative_feats.append(bbox_feats)
    
Negative_feat = np.array(Negative_feats)

feat_train = np.concatenate((Positive_feat,Negative_feat), axis = 0)   


# In[76]:


# Logistic Regression
dtype = torch.DoubleTensor 

r = 1e-10
n = 100
NN = len(y_train)
beta = 0.9

for t in range(500*30):

    i = np.random.choice(range(0,NN), size=n, replace=False)

    x = Variable(torch.from_numpy(feat_train[i]).type(dtype), 
                      requires_grad=False)   

    y_pred = (x.view(n,11776)).mm(w1)

    y_original = Variable(torch.from_numpy(y_train[i]).type(dtype), 
                          requires_grad=False)

    loss = lambda1/2*w1.pow(2).sum() +            (y_original.view(n,1)*torch.log(1+torch.exp(-y_pred)) +                  (1-y_original.view(n,1))*torch.log(1+torch.exp(y_pred))).sum()/n
        
       # Print the training loss on random 2000 data points
    if t%500 ==0:
        S = 2000
        if S>NN_val:
            S = NN_val
        t = np.random.choice(range(0,NN), size=S, replace=False)
        x_2000 = Variable(torch.from_numpy(feat_train[t]).type(dtype), 
                      requires_grad=False)
        y_2000 = Variable(torch.from_numpy(y_train[t]).type(dtype), 
                          requires_grad=False)
               
        y_pred_2000 = (x_2000.view(S,11776)).mm(w1)
        gap_all = y_2000 - y_pred_2000
        
        loss_all = lambda1/2*w1.pow(2).sum()+            (y_2000.view(S,1)*torch.log(1+torch.exp(-y_pred_2000)) +                  (1-y_2000.view(S,1))*torch.log(1+torch.exp(y_pred_2000))).sum()/S
            
        losses_train_linear.append(loss_all.data[0])
        print(loss_all.data[0])
        
        y_pred_2000_pro = 1/(1+torch.exp(-y_pred_2000))
        
        precision = average_precision_score(np.array(y_2000.data), 
                                            np.array(y_pred_2000_pro.data),
                                            average='micro')
        AP_train_linear.append(precision)
        print(precision)


        ## Validation

        t_val = np.random.choice(range(0,NN_val), size=S, replace=False)
        
        x_2000_val = Variable(torch.from_numpy(feat_val[t_val]).type(dtype), 
                      requires_grad=False)
        y_2000_val = Variable(torch.from_numpy(y_val[t_val]).type(dtype), 
                          requires_grad=False)
               
        y_pred_2000_val = (x_2000_val.view(S,11776)).mm(w1)
        
        gap_all_val = y_2000_val - y_pred_2000_val
        
        loss_all_val = lambda1/2*w1.pow(2).sum()+            (y_2000_val.view(S,1)*torch.log(1+torch.exp(-y_pred_2000_val)) +                  (1-y_2000_val.view(S,1))*torch.log(1+torch.exp(y_pred_2000_val))).sum()/S
            
        losses_val_linear.append(loss_all_val.data[0])
        
        print(loss_all_val.data[0])
        
        y_pred_2000_pro_val = 1/(1+torch.exp(-y_pred_2000_val))
        
        precision_val = average_precision_score(np.array(y_2000_val.data), 
                                            np.array(y_pred_2000_pro_val.data),
                                            average='micro')
        AP_val_linear.append(precision_val)
        print()

    loss.backward()
    
    z1= beta*z1+w1.grad.data
    w1.data -= r *z1
   
    w1.grad.data.zero_()


# In[77]:


plt.figure(figsize = (10,12))
plt.plot(AP_train_linear[:230], label = "Train")
plt.plot(AP_val_linear[:230], label = "Val")
plt.title("AP for Category {}".format(cat_id_to_name.get(C[category])), fontsize = 20)
plt.legend(fontsize = 20)
plt.ylabel("AP", fontsize = 20)
plt.xlabel("Epoch", fontsize = 20)


# In[78]:


plt.figure(figsize = (10,12))
plt.plot(losses_train_linear[:230], label = "Train")
plt.plot(losses_val_linear[:230], label = "Val")

plt.title("Loss for Category {}".format(cat_id_to_name.get(C[category])), fontsize = 20)
plt.legend(fontsize = 20)
plt.ylabel("Loss", fontsize = 20)
plt.xlabel("Epoch", fontsize = 20)


# In[79]:


## Test Set

X = Variable(torch.from_numpy(feat_test).type(dtype), 
                  requires_grad=False)   

Y_pred = X.mm(w1)

Y_original = Variable(torch.from_numpy(y_test).type(dtype), 
                      requires_grad=False)


# In[80]:


(y_pro_test >= 0.5 ).sum()


# In[81]:


y_pro_test = 1/(1+torch.exp(-Y_pred))

AP_score = average_precision_score(np.array(Y_original.data), 
                                    np.array(y_pro_test.data),
                                    average='micro')


# In[82]:


AP_score


# In[2307]:


AP_test.append(AP_score )


# In[2308]:


AP_test


# In[1320]:


# Ws = []


# In[1631]:


Ws.append(w1)


# In[1632]:


Ws

