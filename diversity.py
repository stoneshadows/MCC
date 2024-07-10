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
from deephunter.models.alexnet import AlexNet
from deephunter.models.covnet_mnist import ConvnetMnist
from deephunter.models.covnet_cifar10 import ConvnetCifar
from deephunter.models.vgg_cifar10 import VGG16
import pickle
from tqdm import tqdm
import random
import Adversarial_Attack as AA
from collections import Counter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 1
threshold = 0.8

def matrix(dic_up, dic_down):
    len1 = len(dic_up)
    len2 = len(dic_down)  

    keys1 = list(dic_up.keys())
    keys2 = list(dic_down.keys())  

    trans_matrix = np.zeros((len1, len2))  


    for i in range(len1):
        for j in range(len2):
            num = 0
            list1 = dic_up.get(keys1[i])
            list2 = dic_down.get(keys2[j])

            if len(list1) == 0:
                trans_matrix[i][j] = 0
            else:
                for case in list1:
                    if case in list2:
                        num = num + 1
                trans_matrix[i][j] = num / len(list1)
    return trans_matrix


def E_dis(matrix_train, matrix_test):
    dis = 0
    for i in range(len(matrix_train)):
        for j in range(len(matrix_train[i])):
            dis = dis + (matrix_train[i][j] - matrix_test[i][j]) * (matrix_train[i][j] - matrix_test[i][j])
    return dis ** 0.5

def C_dis(matrix_train, matrix_test):
    dis = 0
    for i in range(len(matrix_train)):
        for j in range(len(matrix_train[i])):
            if matrix_test[i][j] == 0 or matrix_train[i][j] == 0:
                continue
            # elif matrix_train[i][j] != 0 and matrix_test[i][j] == 0:
            #     # dis = dis - 1
            #     continue
            else:
                dis = dis + (matrix_train[i][j] * np.log(matrix_train[i][j]/matrix_test[i][j]))

    return dis


def test_states(paths, neurons_states, samples_class, picked_units, num_layers):
    test_state_dict = [{} for _ in range(num_layers)]
    for layer in range(num_layers):
        keys_list = list(neurons_states[layer].keys())   
        test_state_dict[layer] = dict([(key, []) for key in keys_list])

    for id in range(len(samples_class)):
        n_state = [[] for _ in range(num_layers)]
        for layer in range(num_layers):
            n = paths[id][layer]

            for n_i in n:
                if n_i in picked_units[layer]:  
                    n_state[layer].append(n_i)  


            n_state[layer].sort()
            layer_state = neurons_states[layer]   
            keys_list = list(layer_state.keys())

            if str(n_state[layer]) in keys_list:    
                test_state_dict[layer][str(n_state[layer])].append(id)  

    return test_state_dict


def entropy_coverage(neurons_states, test_states, layer):
    dis = []
    ent = []
    # trans_list_train = []
    # trans_list_test = []

    numall = [0 for _ in range(layer)]
    for i in range(layer):
        if i != layer:
            dic1 = neurons_states[i]
            numall[i] = len(dic1)
            dic2 = neurons_states[i + 1]
            ma1 = matrix(dic1, dic2)
            # print(ma1)
            # trans_list_train.append(matrix(dic1, dic2))
            dic11 = test_states[i]
            dic22 = test_states[i + 1]
            ma2 = matrix(dic11, dic22)
            # print(ma2)
            # trans_list_test.append(matrix(dic11, dic22))
            c_dis = C_dis(ma1, ma2)
            e_dis = E_dis(ma1, ma2)
            dis.append(c_dis)
            ent.append(e_dis)

    new_list = [n / sum(numall) for n in numall]
    dis_w = np.multiply(np.array(dis), np.array(new_list))
    ent_w = np.multiply(np.array(ent), np.array(new_list))
    return dis, dis_w, ent, ent_w


def get_sc(lower, upper, k, sa):
    buckets = np.digitize(sa, np.linspace(lower, upper, k))
    return len(list(set(buckets))) / float(k) * 100, buckets

def dist(cov_list):
    cov_sum_list = []
    for i in range(len(cov_list)):
        cov_sum_list.append(sum(cov_list[i])/len(cov_list[i]))  
    return cov_sum_list

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


def generate_class(testset):
    classes = [[] for i in range(10)]
    for idx, (image, label) in enumerate(testset):
        classes[label].append(idx)

    size = 100
    allc = [i for i in range(10)]
    had = []
    k = 1
    test_set = []
    all_class_set = []
    for m in range(len(classes)):
        while len(had) <= m:
            leave = list(set(allc) - set(had))
            cl = random.sample(leave, 1)[0]
            test_set.append(classes[cl])
            had.append(cl)
            k += 1

        test_set1 = []
        for ccla in test_set:
            s_data = random.sample(ccla, int(size//(m+1)))
            test_set1.extend(s_data)

        if len(test_set1) < size:
            flat_cla = [item for sublist in test_set for item in sublist]
            lea_list = list(set(flat_cla) - set(test_set1))
            leamin = random.sample(lea_list, (size - len(test_set1)))
            test_set1.extend(leamin)

        all_class_set.append(test_set1)
    return all_class_set

if daset == 'svhn':
   testset = torchvision.datasets.SVHN(root='./data/',
                                           transform=transforms.ToTensor(), split='test')
   path_train = "./cluster_paths/alexnet/trains/"
   feature_size = [64, 192, 384, 256, 256]
   model = AlexNet()
   model.load_state_dict(torch.load("./data/trained_models/alexnet.pkl"))
elif daset == 'mnist':
    testset = torchvision.datasets.MNIST(root='./data/',
                                           transform=transforms.ToTensor(), train=False)
    path_train = "./cluster_paths/convmnist/trains/"
    model = ConvnetMnist()
    model.load_state_dict(torch.load("./data/trained_models/mnist.pth")["net"])
    feature_size = [64, 64, 128]
elif daset == 'cifar10conv':
    testset = torchvision.datasets.CIFAR10(root='./data/', transform=transforms.ToTensor(), train = False,
                                download=True)
    feature_size = [32, 32, 64, 64, 128, 128, 1024, 512]
    path_train = "./cluster_paths/convcifar10/trains/"
    model = ConvnetCifar()
    model.load_state_dict(torch.load("./data/trained_models/cifar.pth")["net"])

elif daset == 'cifar10':
    testset = torchvision.datasets.CIFAR10(root='./data/', transform=transforms.ToTensor(), train = False,
                                download=True)
    feature_size = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    model = VGG16()
    model.load_state_dict(torch.load("./data/trained_models/vgg.pkl"))
    path_train = "./cluster_paths/vgg/trains/" 

    from torchvision import models
    model_save_path = './data/trained_models/ResNet.pth'
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    model.load_state_dict(torch.load(model_save_path))


ori_model = model.cuda()
ori_model.eval()


# elif args.arc == "resnet18":
#     feature_size = [64, 64, 64, 128, 128, 256, 256, 512, 512]



num_layers = len(feature_size)

dis1,disw1, ent1, entw1 = [], [], [], []
ent_list = []
for _ in range(20):
    all_class_set = generate_class(testset)
    for att, cla_set in enumerate(all_class_set):
        print(att)
        attack_name = att
        samples_class = [[] for c in range(10)]
        right_samples_class = [[] for c in range(10)]
        paths_class = [[] for c in range(10)]

        selected_data = [testset[i] for i in cla_set]


        data_loader = Data.DataLoader(dataset= selected_data, batch_size=batch_size, shuffle=False)

        for i, (data, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
            data, target = data.cuda(), target.cuda()
            batch_size = int(data.size()[0])

            # if i in attacklist:
            model_prediction, _, true_relevance = inn_model.innvestigate(in_tensor=data)
            true_relevance = true_relevance[:]

            # if i == min(attacklist):
            if i == 0:  # 需要修改 if i == attacklist[0]:
                relev = true_relevance
            else:
                for l in range(len(relev)):
                    relev[l] = torch.cat((relev[l], true_relevance[l]), 0)

            torch.save(relev, root_path + "{}_relev_test.pt".format(attack_name))

        e1_time = time.time()

        num_layers = len(relev) - 1
        sample_neurons = {}

        relev = torch.load(root_path + "{}_relev_test.pt".format(attack_name))

        for layer in range(len(relev)):
            true_layer = num_layers - layer
            r = relev[true_layer]
            units, rests, sec_cris = pick_neurons_layer(r, threshold, False)  # extract paths
            for i in range(len(units)):
                if layer == 0:
                    sample_neurons[i] = []
                sample_neurons[i].append(units[i])

        # save_path = root_path + "{}_lrp_path_threshold{}_test.pkl".format(attack_name, threshold)
        # output = open(save_path, 'wb')
        # pickle.dump(sample_neurons, output)
        # print('test paths save!')

        ######################################################
        paths = sample_neurons

        num_layers = len(feature_size)
        neurons = [[] for _ in range(num_layers)]
        neurons_state = [[] for _ in range(num_layers)]
        sensUnits = [[] for _ in range(num_layers)]
        sec_picked_units = [[] for _ in range(num_layers)]
        sta = [[] for _ in range(num_layers)]


        for index in range(len(cla_set)):
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
                    if s < round(len(cla_set) * threshold):  
                        picked_units[layer] = sensUnits[layer][:t + 1]
                        if t * 2 < len(sensUnits[layer]):  
                            sec_picked_units[layer] = sensUnits[layer][t: t * 2]  
                        else:
                            sec_picked_units[layer] = sensUnits[layer][t:]
                        break

                    if t == len(sta[layer]) - 1:
                        # print("warning")
                        picked_units[layer] = sensUnits[layer]

        # path_fname = root_path + "threshold{}_paths.pkl".format(threshold)
        # output = open(path_fname, 'wb')
        # pickle.dump(picked_units, output)
        #
        # print('picked_units save!')



        merged_dict = [{} for _ in range(num_layers)]
        for cla in range(10):
            neurons_path = path_train
            states = neurons_path + "neurons_states_class{}.pkl".format(cla)   
            with open(states, 'rb') as fr:
                neurons_states = pickle.load(fr)

            for lay in range(num_layers):
                for key, value in neurons_states[lay].items():
                    merged_dict[lay].setdefault(key, []).extend(value)

        neurons_states = merged_dict
        for lay in range(num_layers):
            max_length_element = max(neurons_states[lay].keys(), key=len)




        test_state = test_states(paths, neurons_states, cla_set, picked_units, num_layers)
        dis, disw, ent, entw = entropy_coverage(neurons_states, test_state, num_layers-1) 

        # print(dis)
        print(ent)

        # print("Dis:", sum(dis))
        # print("Disw:", sum(disw))
        print("Ent:", sum(ent))
        # print("Entw:", sum(entw))

        dis1.append(sum(dis))
        disw1.append(sum(disw))
        ent_list.append(ent)
        entw1.append(sum(entw))
        ent1.append(sum(ent))

np.save(root_path + 'ent_list.npy', ent_list)
np.save(root_path + 'dis1.npy'.format(fid), dis1)
np.save(root_path + 'disw1.npy'.format(fid), disw1)
np.save(root_path + 'ent1.npy'.format(fid), ent1)
np.save(root_path + 'entw1.npy'.format(fid), entw1)


