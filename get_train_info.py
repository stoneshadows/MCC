
import pickle
import torch
import numpy as np
import torchvision
import sys
# sys.path.append("models")
import os
import torchvision.transforms as transforms

from collections import Counter
from torch.utils.data import Subset

import torchvision.datasets as datasets
import random
import torch.utils.data as Data
from sklearn.cluster import KMeans
import numpy as np
import argparse
from deephunter.models.alexnet import AlexNet
from deephunter.models.covnet_mnist import ConvnetMnist
from deephunter.models.covnet_cifar10 import ConvnetCifar
from deephunter.models.vgg_cifar10 import VGG16

# from models.sa_models import ConvnetMnist, ConvnetCifar
# from AlexNet_SVHN import AlexNet
# from vgg import vgg16_bn

from deephunter.models import get_net,get_masked_net #  .sa_models import mask_ConvnetMnist, mask_ConvnetCifar
# from model_mask_vgg import mask_VGG16 # imagenet's vgg
# from mask_vgg import mask_vgg16_bn
from deephunter.models.masked.mask_AlexNet_SVHN import mask_AlexNet


# from imagenet10Folder import imagenet10Folder # dataset for imagenet

parser = argparse.ArgumentParser(description='model interpretation')
parser.add_argument('--paths_path', type=str, default="./cluster_paths_0/SVHN_alexnet_lrp_path_threshold0.8_train.pkl")
parser.add_argument('--arc', type=str, default="alexnet")
parser.add_argument('--data_train', action='store_true')
parser.add_argument('--b_cluster', action='store_true')
parser.add_argument('--useOldCluster', action='store_true')
parser.add_argument('--dataset', type=str, default="svhn")
parser.add_argument('--attack', type=str, default="")
parser.add_argument('--gpu', type=str, default="0")
# parser.add_argument('--n_clusters', type=int, default=3)
parser.add_argument('--grids', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--threshold', type=float, default=0.8)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from tqdm import tqdm

batch_size = args.batch_size
     
data_path = args.paths_path
with open(data_path, 'rb') as fr:
    paths = pickle.load(fr)




if args.dataset == "mnist":
    ori_model = ConvnetMnist()
    ori_model.load_state_dict(torch.load("./data/trained_models/mnist.pth")["net"])

elif args.dataset == "svhn":
    dataset = torchvision.datasets.SVHN(root='.data/', split='train',
                                        transform=transforms.ToTensor(), download=True)
    data_loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    ori_model = AlexNet()
    ori_model.load_state_dict(torch.load("./data/trained_models/alexnet.pkl"))

elif args.dataset == "cifar10":
    dataset = torchvision.datasets.CIFAR10(root='./data/', transform=transforms.ToTensor(),
                                           train=True,
                                           download=True)
    data_loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    # ori_model = ConvnetCifar()
    # ori_model.load_state_dict(torch.load("./data/trained_models/cifar_mixup_acc_90.36_ckpt.pth")["net"])
    # ori_model = VGG16()
    # ori_model.load_state_dict(torch.load("./data/trained_models/vgg_seed32_dropout.pkl"))
ori_model = ori_model.cuda()
ori_model.eval()

if args.arc == "vgg" or args.arc == "vgg16_bn":
    feature_size = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
elif args.arc == "resnet18":
    feature_size = [64, 64, 64, 128, 128, 256, 256, 512, 512]
elif args.arc == "convmnist":
    feature_size = [64, 64, 128]
elif args.arc == "convcifar10":
    feature_size = [32, 32, 64, 64, 128, 128, 1024, 512]
elif args.arc == "alexnet":
    feature_size = [64, 192, 384, 256, 256]
    

samples_class = []
right_samples_class = []
paths_class = []
flatten_paths = []
binary_paths = []

root_path = "./cluster_paths/{}/trains/".format(args.arc)
if os.path.exists(root_path) == False:
    os.makedirs(root_path)
samples_class_file = root_path + "samples_class_{}.pkl".format(args.dataset)
right_samples_class_file = root_path + "right_samples_class_{}.pkl".format(args.dataset)

if os.path.exists(samples_class_file):
    with open(samples_class_file, "rb") as f:
        unpickler = pickle.Unpickler(f)
        samples_class = unpickler.load()
    with open(right_samples_class_file, "rb") as f:
        unpickler = pickle.Unpickler(f)
        right_samples_class = unpickler.load()
else: 
    start_index = end_index = 0
    for step, (val_x, val_y) in tqdm(enumerate(data_loader), total=len(data_loader)):
        val_x = val_x.cuda()
        start_index = end_index
        # print("step:", step)
        val_output = ori_model(val_x)
        _, val_pred_y = val_output.max(1)
        end = False
        for i, t in enumerate(val_pred_y):
            if val_y[i] >= 10:
                end = True
                break
            if t < 10:
                samples_class.append(i+start_index)
            if t == val_y[i]:
                right_samples_class.append(i+start_index)
        if end:
            break
        end_index = start_index + val_x.shape[0]
     

# samples_class_all = samples_class
# print('samples_class_all', len(samples_class_all))

num_layers = len(feature_size)


neurons = [[] for _ in range(num_layers)]
neurons_state = [[] for _ in range(num_layers)]
sensUnits = [[] for _ in range(num_layers)]
sta = [[] for _ in range(num_layers)]
state_dict = [{} for _ in range(num_layers)]  


picked_units = [[] for _ in range(num_layers)]
rest_units = [[] for _ in range(num_layers)]
sec_picked_units = [[] for _ in range(num_layers)]

for index in range(len(paths)):
    for layer in range(num_layers):
        n = paths[index][layer]
        neurons[layer].extend(n) 


    for layer in range(num_layers):
        neurons[layer].extend([i for i in range(feature_size[layer])])  

    for layer in range(num_layers):
        sens = []
        s = []
        c = Counter(neurons[layer])
        mc = c.most_common()  
        for a, b in mc:  
            sens.append(a)
            s.append(b)
        sensUnits[layer] = sens
        sta[layer] = s  
     

    picked_units = [[] for _ in range(num_layers)]
    for layer in range(num_layers):
        for t, s in enumerate(sta[layer]):
            if s < round(len(paths) * args.threshold):  
                picked_units[layer] = sensUnits[layer][:t + 1]
                if t * 2 < len(sensUnits[layer]): 
                    sec_picked_units[layer] = sensUnits[layer][t: t * 2]  
                else:
                    sec_picked_units[layer] = sensUnits[layer][t:]
                break

            if t == len(sta[layer]) - 1:
           
                picked_units[layer] = sensUnits[layer]


for index in range(len(paths)):
    n_state = [[] for _ in range(num_layers)]
    for layer in range(num_layers):
        n = paths[index][layer]  
        for n_i in n:
            if n_i in picked_units[layer]: 
                n_state[layer].append(n_i)  
        n_state[layer].sort()

        state_dict[layer].setdefault('{}'.format(n_state[layer]), []).append(index)  

path_fname1 = root_path + "neurons_states.pkl"
output1 = open(path_fname1, 'wb')
pickle.dump(state_dict, output1)

















