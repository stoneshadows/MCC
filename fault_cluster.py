
import torchvision
import sys
# sys.path.append("models")
import os
import torchvision.transforms as transforms

from collections import Counter
from torch.utils.data import Subset
from torchvision import models
import torchvision.datasets as datasets
import random
import torch.utils.data as Data
from sklearn.cluster import KMeans
import numpy as np
import argparse
from models.alexnet import AlexNet


from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import hdbscan
import numpy as np
import time
import copy
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import umap.umap_ as umap

random.seed(1024)
np.random.seed(1024)

batch_size = 32
     
# data_path = args.paths_path
# with open(data_path, 'rb') as fr:
#     paths = pickle.load(fr)

fid = 'results'
daset = 'alexnet'
root_path = "./{}/{}/".format(fid, daset)

CLIP_MIN = -0.5
CLIP_MAX = 0.5

resize_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

transformcifar10 = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

def scale_one(X):
    nom = (X - X.min()) * (1)
    denom = X.max() - X.min()
    koko = nom / denom
    return koko

plot_kwds = {'alpha': 0.15, 's': 80, 'linewidths':0}
def plot_clusters(data, algorithm, args, kwds):
    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    print("labels", labels)
    print("max", labels.max())
    global ll
    ll=copy.deepcopy(labels)
    # ll.sort()
    # print(ll)
    kk=list(copy.deepcopy(ll))
    goo=copy.deepcopy(ll)
    goo.sort()

    i=-1
    while (i<= goo.max()):
      k=kk.count(i)
      i = i + 1
    end_time = time.time()

    #
    classified_lists = {}
    for index, value in zip(labels, fea_lab):
        if index not in classified_lists:
            classified_lists[index] = []
        classified_lists[index].append(value)

    final_clus = []
    for index, classified_list in classified_lists.items():
        final_clus.append(classified_list)
    print(final_clus)
    np.save(root_path +'fault_cluster2.npy', final_clus)
    return labels




testset = torchvision.datasets.SVHN(root='./data/',
                                       transform=transforms.ToTensor(), split='test')
trainset = torchvision.datasets.SVHN(root='./data/',
                                       transform=transforms.ToTensor(), split='train')

x_test1 = testset.data
x_train1 = trainset.data


path_train = "./alexnet/trains/"
feature_size = [64, 192, 384, 256, 256]
model = AlexNet()
model.load_state_dict(torch.load("./data/trained_models/alexnet.pkl"))


data_loader = Data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)
data_loader1 = Data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=False)


ori_model = model.cuda()
ori_model.eval()


total_times = 40
total = 0
accuracy_rate = []


start_index = end_index = 0
fea_lab = []
valy = []
val_pred_ys = []
f = 0
for step, (val_x, val_y) in tqdm(enumerate(data_loader), total=len(data_loader)):
    val_x = val_x.cuda()
    start_index = end_index
    val_output = ori_model(val_x)
    _, val_pred_y = val_output.max(1)

    end = False
    for i, t in enumerate(val_pred_y):
        if val_y[i] >= 10:
            end = True
            break
        if t != val_y[i]:
            fea_lab.append(f)
            valy.append(val_y[i])
            val_pred_ys.append(val_pred_y[i])
        f += 1

    if end:
        break
    end_index = start_index + val_x.shape[0]

print(len(fea_lab))
testsubset = torch.utils.data.Subset(testset, fea_lab)


########

start_index = end_index = 0
fea_labtr = []
valy = []
val_pred_ys = []
f1 = 0
for step, (val_x, val_y) in tqdm(enumerate(data_loader1), total=len(data_loader1)):
    val_x = val_x.cuda()
    start_index = end_index
    val_output = ori_model(val_x)
    _, val_pred_y = val_output.max(1)

    end = False
    for i, t in enumerate(val_pred_y):
        if val_y[i] >= 10:
            end = True
            break
        if t != val_y[i]:
            fea_labtr.append(f1)
            valy.append(val_y[i])
            val_pred_ys.append(val_pred_y[i])
        f1 += 1

    if end:
        break
    end_index = start_index + val_x.shape[0]
print(len(fea_labtr))
trainsubset = torch.utils.data.Subset(trainset, fea_labtr)

x_test_resized = [resize_transform(image) for image,_ in testsubset]
x_train_resized = [resize_transform(image) for image,_ in trainsubset]


x_test1_resized = torch.stack(x_test_resized)

x_test1_normalized = (x_test1_resized / 255.0) - (1.0 - CLIP_MAX)

vgg16 = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
# Freeze the pretrained layers
for param in vgg16.parameters():
    param.requires_grad = False
name_layer = 'features.24'  # Corresponds to 'block5_conv3'
intermediate_layer_model = nn.Sequential(*list(vgg16.features.children())[:25])  # Up to block5_conv3
intermediate_layer_model.eval()
with torch.no_grad():
    features = intermediate_layer_model(x_test1_normalized)

features = features.view(features.size(0), -1)
nom = (features - features.min(dim=0)[0]) * (1 - 0)
denom = features.max(dim=0)[0] - features.min(dim=0)[0]
denom[denom == 0] = 1
X_scf = nom / denom


X_features_stacked = X_scf
X_features = X_features_stacked.numpy()
# X_features = X_features_numpy.transpose()
print(len(X_features))

PY_scaled = scale_one(torch.tensor(val_pred_ys))
TY_scaled = scale_one(torch.tensor(valy))

#####进行faults聚类
kk=[]
trace=[]
ss_tt=[]
d_scale_umap=[]
bb=[]
hdbscan_in_umap=[]
silu=[]
Sumn=0

# for i,j in zip([500,400,300,250],[450,350,250,200]):
#   for k,o in zip([5,10,15,20,25],[3,5,10,15,20]):
#     for n_n in [0.03, 0.1, 0.25, 0.5]:
for i,j in zip([250],[200]):
  for k,o in zip([5],[3]):
    for n_n in [0.05]:
      fit = umap.UMAP(min_dist=n_n, n_neighbors=k)
      u1 = fit.fit_transform(X_features)
      fit = umap.UMAP(min_dist=0.1, n_neighbors=o)
      u = fit.fit_transform(u1)
      print("u",u.shape)
      # for gg in [3,5,10,15,20,30,35,40]:
      print("len", len(list(bb)))
      labels = plot_clusters(u, hdbscan.HDBSCAN, (), {'min_cluster_size':5})
      # print("noisy", list(ll).count(-1))
      jo=sklearn.metrics.silhouette_score(u, ll)
      print(jo)
      #x_mis standard
      ss=sklearn.metrics.silhouette_score(X_features, ll)
      print(ss)


      # print("x_stand",ss)
      # if ((jo>=0.1 or ss>=0.1) and ll.max()+2>=200):
      #   bb.append(ll)
      #   my_trace = [i,j,k,o,jo,ll.max()+2,list(ll).count(-1)]
      #   trace.append(my_trace)
      # # print("ll",ll)
      #   hdbscan_in_umap.append(u)
      #   print("OOOOOOOOOOOOOO",Sumn)
      #   print("SIIIIIIIIIII",jo , ss)
      #   Sumn=Sumn+1
      #   print("BBBBBBBBBBBBBBBB",list(ll).count(-1))



