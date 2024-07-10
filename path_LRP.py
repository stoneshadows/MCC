# python path_LRP.py --gpu 0 --arc vgg16_bn --threshold 0.7 --dataset imagenet --suffix imagenet --data_train
import numpy as np
import sys
sys.path.append(".")

from LSA_DSA_ANPC_lib.LRP_path.innvestigator import InnvestigateModel
from LSA_DSA_ANPC_lib.LRP_path.inverter_util import Flatten
import time
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
from deephunter.models.sa_models import ConvnetMnist
from deephunter.models.alexnet import AlexNet
from deephunter.models.covnet_mnist import ConvnetMnist
from deephunter.models.covnet_cifar10 import ConvnetCifar
from deephunter.models.vgg_cifar10 import VGG16

import pickle
from tqdm import tqdm
import torchvision.models as models
from torchsummary import summary
import ssl
from deephunter.models.Resnet import ResNet18
import torchvision.models as models

ssl._create_default_https_context = ssl._create_unverified_context
parser = argparse.ArgumentParser(description='get the paths')
parser.add_argument('--gpu', type=str, default="0")
parser.add_argument('--arc', type=str, default="resnet")
parser.add_argument('--threshold', type=float, default=0.8)
parser.add_argument('--dataset', type=str, default="cifar")
parser.add_argument('--last', type=bool, default=False)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

batch_size = 32

def pick_neurons_layer(relev, threshold=0.1, last=False):
    if len(relev.shape) != 2:
        rel = torch.sum(relev, [2, 3])
    else:
        rel = relev
    units_all = []
    rest_all = []
    sec_cri_all = []
    for i in range(rel.shape[0]):
        rel_i = rel[i]
        values, units = torch.topk(rel_i, rel_i.shape[0])
        sum_value = 0
        tmp_value = 0

        part = 0
        if not last:
            for v in values:
                if v > 0:
                    sum_value += v
            for i, v in enumerate(values):
                tmp_value += v
                if tmp_value >= sum_value * threshold or v <= 0:
                    part = i
                    break
            units_picked = units[:part+1].tolist()
            rest = units[part+1:].tolist()
            if part * 2 >= len(units):
                sec_cri = units[part:].tolist()
            else:
                sec_cri = units[part:part*2].tolist()
        else:
            for v in values:
                if v < 0:
                    part = i
            units_picked = units[part:].tolist()
            rest = units[:part].tolist()
        units_all.append(units_picked)
        rest_all.append(rest)
        sec_cri_all.append(sec_cri)
    return units_all, rest_all, sec_cri_all


# testset = torchvision.datasets.SVHN(root='./data/',split='train',
#                                transform=transforms.ToTensor(), download=True)

# testset = torchvision.datasets.MNIST(root='./data/', transform=transforms.ToTensor(), train = True,
#                             download=True)

testset = torchvision.datasets.CIFAR10(root='./data/', transform=transforms.ToTensor(), train = True,
                            download=True)

data_loader = Data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)

with torch.no_grad():

    # model = AlexNet()
    # model.load_state_dict(torch.load("./data/trained_models/alexnet.pkl"))

    # model = ConvnetMnist()
    # model.load_state_dict(torch.load("./data/trained_models/mnist.pth"))

    # model = ConvnetCifar()
    # model.load_state_dict(torch.load("./data/trained_models/cifar.pth"))

    # model = VGG16()
    # model.load_state_dict(torch.load("./data/trained_models/vgg.pkl"))
    #
    # model = model.cuda()
    # model.eval()
    # inn_model = InnvestigateModel(model, lrp_exponent=2,
    #                               method="b-rule",
    #                               beta=.5)

    # resnet_model = models.resnet50(pretrained=True)  
    model_save_path = 'D:/my_code2/NPC-master/data/trained_models/ResNet_CIFAR.pth'
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    model.load_state_dict(torch.load(model_save_path))
    resnet_model = model.cuda()
    # model = model.to(device)
    resnet_model.eval()  


    inn_model = InnvestigateModel(resnet_model, lrp_exponent=2,
                                  method="b-rule",
                                  beta=.5)

    s_time = time.time()

    p = "LRP_path/relevs/{}_relev_{}_train.pt".format(args.dataset, args.arc)
    start = -1
    if os.path.exists(p):
        start, relev = torch.load(p)
        # print("load relev from {}".format(start))


    for i, (data, target) in tqdm(enumerate(data_loader), desc="innvestigate the relevence by LRP.innvestigate",total=len(data_loader)):


        if target[0] < 10:
            # print("index:", i)

            data, target = data.cuda(), target.cuda()
            batch_size = int(data.size()[0])

            model_prediction, _, true_relevance = inn_model.innvestigate(in_tensor=data)

            true_relevance = true_relevance[:]
            # print(true_relevance)
#             print(true_relevance[0].shape)

#             tmp = torch.sum(true_relevance[0].squeeze(), [1, 2])
            if i == 0:
                relev = true_relevance

            else:
                for l in range(len(relev)):
                    relev[l] = torch.cat((relev[l], true_relevance[l]), 0)


        # print("done")
        # print(len(relev[0]))

        torch.save(relev, "{}_relev_{}_train.pt".format(args.dataset, args.arc))


    e1_time = time.time()
    num_layers = len(relev) - 1

    sample_neurons = {}
    sample_rests = {}
    sample_sec = {}
    for layer in range(len(relev)):
        true_layer = num_layers - layer
        r = relev[true_layer]
        units, rests, sec_cris = pick_neurons_layer(r, args.threshold, args.last)
        for i in range(len(units)):
            if layer == 0:
                sample_neurons[i] = []
                sample_rests[i] = []
                sample_sec[i] = []
            sample_neurons[i].append(units[i])
            sample_rests[i].append(rests[i])
            sample_sec[i].append(sec_cris[i])


    save_path = "./cluster_paths_0/{}_{}_lrp_path_threshold{}_train.pkl".format(args.dataset, args.arc, args.threshold)
    save_path_rest = "./cluster_paths_0/{}_{}_lrp_path_threshold{}_train_rest.pkl".format(args.dataset, args.arc, args.threshold)
    save_path_sec = "./cluster_paths_0/{}_{}_lrp_path_threshold{}_train_sec.pkl".format(args.dataset, args.arc, args.threshold)


    e_time = time.time()
    print("relev_time:", e1_time-s_time)
    print("time:", e_time-s_time)

    output = open(save_path, 'wb')
    pickle.dump(sample_neurons, output)
    output_rest = open(save_path_rest, 'wb')
    pickle.dump(sample_rests, output_rest)
    output_sec = open(save_path_sec, 'wb')
    pickle.dump(sample_sec, output_sec)
    print("done")
