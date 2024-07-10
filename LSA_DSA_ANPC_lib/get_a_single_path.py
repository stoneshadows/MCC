import numpy as np
import sys

sys.path.append("../")

from LSA_DSA_ANPC_lib.LRP_path.innvestigator import InnvestigateModel
from LSA_DSA_ANPC_lib.LRP_path.inverter_util import Flatten
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
#from models.VGG_16 import VGG16
# from models.vgg import vgg16_bn
#from models.sa_models import ConvnetMnist, ConvnetCifar
import torchvision.models as models
import pickle


def getPath(data, model, width_threshold, other_label=False, target=None, arc="alexnet"):

    def group_neurons(relev, threshold=0.8, last=False):
        if len(relev.shape) != 2:
            rel = torch.sum(relev, [2, 3])
        else:
            rel = relev
        units_1 = []
        units_2 = []
        units_3 = []
        for i in range(rel.shape[0]):  # 0-7
            rel_i = rel[i]
            values, units = torch.topk(rel_i, rel_i.shape[0])  # 将每一层的神经元按相关值排序；units是序号
            ################################# group and weight
            u_1 = units[0:int(len(units)/3)].tolist()  # high
            u_2 = units[int(len(units) / 3):int(len(units) / 3)*2].tolist()  # medium
            u_3 = units[int(len(units) / 3)*2:len(units)].tolist()  # low

            units_1.append(u_1)
            units_2.append(u_2)
            units_3.append(u_3)

        return units_1, units_2, units_3


    def pick_neurons_layer(relev, threshold=0.8, last=False):
        if len(relev.shape) != 2:
            rel = torch.sum(relev, [2, 3])
        else: 
            rel = relev
        units_all = []
        rest_all = []
        sec_cri_all = []
        for i in range(rel.shape[0]):
            rel_i = rel[i]
            values, units = torch.topk(rel_i, rel_i.shape[0])  # 将每一层的神经元按相关值排序；units是序号

            sum_value = 0
            tmp_value = 0

            part = 0
            if not last:
                for v in values:

                    if v > 0:
                        sum_value += v  # 每一层总的相关值
                for i, v in enumerate(values):
                    tmp_value += v
                    if tmp_value >= sum_value * threshold or v <= 0:  # 相关值达到约束时
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
                units_picked = units[part:] .tolist()
                rest = units[:part].tolist()
            units_all.append(units_picked)
            rest_all.append(rest)
            sec_cri_all.append(sec_cri)
        return units_all, rest_all, sec_cri_all


    with torch.no_grad():

        model = model.cuda()
        model.eval()
        # Convert to innvestigate model
        inn_model = InnvestigateModel(model, lrp_exponent=2,
                                      method="b-rule",
                                      beta=.5)
        

        data = data.cuda()
        if not other_label:
            model_prediction, _, true_relevance = inn_model.innvestigate(in_tensor=data)
        else:
            model_prediction, _, true_relevance = inn_model.innvestigate(in_tensor=data, rel_for_class=other_label)
#       print(true_relevance)
        relev = true_relevance[::-1]
        if arc == "alexnet":
            relev = relev[:-1]
        if arc == "vgg16_bn":
            relev = relev[:-2]

        sample_neurons = {}
        for layer in range(len(relev)):

            r = relev[layer]
            units, _, _ = pick_neurons_layer(r, width_threshold)  # 每一层挑选神经元
            for i in range(len(units)):
                if layer == 0:
                    sample_neurons[i] = []
                sample_neurons[i].append(units[i])
        return sample_neurons

        # sample_neurons1 = {}
        # sample_neurons2 = {}
        # sample_neurons3 = {}
        # for layer in range(len(relev)):
        #
        #     r = relev[layer]
        #     units1, units2, units3 = group_neurons(r, width_threshold)
        #     for i in range(len(units1)):
        #         if layer == 0:
        #             sample_neurons1[i] = []
        #         sample_neurons1[i].append(units1[i])
        #     for i in range(len(units2)):
        #         if layer == 0:
        #             sample_neurons2[i] = []
        #         sample_neurons2[i].append(units2[i])
        #     for i in range(len(units3)):
        #         if layer == 0:
        #             sample_neurons3[i] = []
        #         sample_neurons3[i].append(units3[i])
        # return sample_neurons1, sample_neurons2, sample_neurons3



if __name__=="__main__":
    from deephunter.models.alexnet import AlexNet
    dataset = torchvision.datasets.SVHN(root='../deephunter/models/data/',
                                   transform=transforms.ToTensor(), download=True)

    data_loader = Data.DataLoader(dataset=dataset, batch_size=5, shuffle=False)
    model = AlexNet()
    model.load_state_dict(torch.load("../data/trained_models/alexnet_lr0.0001_39.pkl"))
    # model = models.alexnet(pretrained=True) # sy
    model = model.cuda()
    model.eval()
    for i, (data, target) in enumerate(data_loader):
        data = data
        # for index, (name, param) in enumerate(model.named_parameters()):
        #     print(str(index) + " " + name)
        # summary(model, (3, 32, 32))
        the_path = getPath(data, model, width_threshold=0.5, other_label=False, target=None, arc="alexnet")
        print(the_path.keys())
