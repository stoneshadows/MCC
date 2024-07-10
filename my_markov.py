import pickle
import argparse
import numpy as np
import math
import torch
import os
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
import random
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
from deephunter.models.alexnet import AlexNet
from torch.utils.data import RandomSampler
    # from cluster_three_level_mask_Test import test_states

#np.set_printoptions(threshold = 1e6)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                # dis = dis - scipy.stats.entropy(matrix_train[i][j], matrix_test[i][j])
    return dis


def test_states(paths, neurons_states, samples_class, picked_units, num_layers):
    test_state_dict = [{} for _ in range(num_layers)]
    for layer in range(num_layers):
        keys_list = list(neurons_states[layer].keys())     
        test_state_dict[layer] = dict([(key, []) for key in keys_list])
 
    for id in samples_class:  

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


def cov_cal(cov_add):
    covered = 0
    for l in range(layers - 1): 
        buckets = np.digitize(cov_add[l], np.linspace(min(cov_add[l]), max(cov_add[l]), n_bucket))
        new_buckets = [str(l) + "_" + str(buckets[i]) for i in range(len(buckets))]
        # print(set(new_buckets))
        covered += len(set(new_buckets))
    target_cov = covered / (float(n_bucket) * layers * 10) * 100

    # target_cov, buckets = get_sc(min(cov_add), max(cov_add), n_bucket, cov_add)

    results_my[attack_name].append(target_cov)
    print("test_my_score--->", target_cov)
    map_result[attack_name] = target_cov

    return map_result, target_cov

def cov_rate(covall):
    cov_change = [0 for _ in range(len(covall))]
    # #计算变化率
    for i in range(len(covall)):
        max_change = max(covall) - min(covall)
        if max_change != 0:
            cov_change[i] = (covall[i] - covall[0]) / (max(covall) - min(covall))
    return cov_change

if __name__ == '__main__':
    # ma1 = [[1/2,1/2,0], [0,0,1], [2/3,1/3,0],[0,1/3,2/3]]
    # ma2 = [[1/2,1/2,0], [0,0,1], [1/2,1/2,0],[0,2/3,1/3]]
    # e_dis = E_dis(ma1, ma2)
    # c_dis = C_dis(ma1, ma2)
    # print(e_dis, c_dis)
    classes = 10
    # clusters = 3
    batch_size = 32

    # fid = 'adv8'
    # map_arg = 'alexnet'
    # map_dataset = 'svhn'
    # feature_size = [64, 192, 384, 256, 256]
    # layers = len(feature_size)

    # feature_size = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    # layers = len(feature_size)
    # fid = "adv_vgg"
    # map_dataset = 'cifar10'
    # map_arg = 'vgg'

    fid = 'adv_mnist1'
    feature_size = [64, 64, 128]
    layers = len(feature_size)
    map_dataset = 'mnist'
    map_arg = 'convmnist'

    # feature_size = [64, 192, 384, 256, 256]
    # layers = len(feature_size)
    # fid = 'adv_svhn1'
    # map_dataset = 'svhn'
    # map_arg = 'alexnet'

    # feature_size = [32, 32, 64, 64, 128, 128, 1024, 512]
    # layers = len(feature_size)
    # fid = "adv_cifar1"
    # map_dataset = 'cifar10'
    # map_arg = 'convcifar10'

    neurons_path = "./cluster_paths/{}/trains/".format(map_arg)  

    args_attack = 'nature'
    save_dir = "./{}/".format(fid)

    error_rate_list = [0., 1.]
    # error_rate_list = [0.01, .1, .2, .3, .4, .5, .6, .7, .8, .9]
    # error_rate_list = [0.01, .1,  .3,  .7]
    # error_rate_list = [.05, .1, .15, .2, .25, .3, .35, .4, .45, .5]
    # error_rate_list = [.01, .02, .04, .06, .08, 0.1]
    cov_list_dis = [[] for _ in range(len(error_rate_list))]
    cov_list_disw = [[] for _ in range(len(error_rate_list))]
    cov_list_ent = [[] for _ in range(len(error_rate_list))]
    cov_list_entw = [[] for _ in range(len(error_rate_list))]
    e = 0
    covall1 = []
    covall2 = []
    covall3 = []
    covall4 = []

    # dictall = {'MCEC': [], 'MCECw': [],'NC':[], 'KMSC':[],
    #            'MLC':[], 'MLSC':[], 'LSC':[], 'DSC':[], 'ANPC':[], 'SNPC':[]}

    dictall = {'MCEC': [], 'MCECw': []}

    methods = ['NC', 'KMSC', 'MLC', 'MLSC', 'lsa', 'dsa', 'nma', 'intra']

    # methods = ['my']
    cov_list_other = [[] for _ in range(len(error_rate_list))]

    for error_rate in error_rate_list:
        n_bucket = 200
        results_my = {}
        map_result = {}

        attack_name = '{}_{}'.format(args_attack, error_rate)
        results_my[attack_name] = []
        root_path = "./{}/cluster_paths_{}/{}/".format(fid, error_rate, attack_name)

        if args_attack == 'nature':
            # data_path = "./cluster_paths_0/SVHN_alexnet_lrp_path_threshold0.8_train.pkl"  # SVHN
            # data_path = "./{}/cluster_paths_0/PGD_0_lrp_path_threshold0.8_test.pkl".format(fid)
            data_path = "./adv_cifar_vgg/cluster_paths_0/PGD_0_lrp_path_threshold0.8_test.pkl"
            data_path = "./adv_mnist/cluster_paths_0/PGD_0_lrp_path_threshold0.8_test.pkl"
            with open(data_path, 'rb') as fr:
                paths = pickle.load(fr)

        #root_path = "./cluster_paths_{}/{}/".format(error_rate,args.arc)

        #######
        # dataset = torchvision.datasets.SVHN(root='./deephunter/models/data/', split='test',
        #                                transform=transforms.ToTensor(), download=True)
        # data_loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
        #######
        cov_dis = [0]*(layers - 1)
        cov_disw = [0] * (layers - 1)
        cov_ent = [0] * (layers - 1)
        cov_entw = [0] * (layers - 1)


        samples_set = []
        new_sample_set = []

        cov_add_dis = [[] for i in range(layers)]
        cov_add_disw = [[] for i in range(layers)]
        cov_add_ent = [[] for i in range(layers)]
        cov_add_entw = [[] for i in range(layers)]

        total = 0
        print("error_rate:", error_rate)
        for cla in range(classes):
            picked_s = root_path + "samples_class{}.pkl".format(cla)
            with open(picked_s, 'rb') as fp:
                samples_class = pickle.load(fp)  
            # sample_num = random.randint(1, len(samples_class))
            # sample_num = int(len(samples_class) * 0.5)
            # new_sample_class = random.sample(samples_class, sample_num

            # total = total + len(new_sample_class)
            # if total == 100:
            #     pick_list = new_sample_set
            #############
            # picked = root_path + "num_cluster{}_threshold0.8_class{}_cluster{}_paths.pkl".format(clusters,cla, cluster)
            picked = root_path + "threshold0.8_class{}_paths.pkl".format(cla)
            with open(picked, 'rb') as fi:
                picked_u = pickle.load(fi)  

            ####
            # states = root_path + "neurons_states_class{}_cluster{}.pkl".format(cla, cluster)
            states = neurons_path + "neurons_states_class{}.pkl".format(cla)   
            with open(states, 'rb') as fr:
                neurons_states = pickle.load(fr)

            # samples_class = new_sample_class

            test_state = test_states(paths, neurons_states, samples_class, picked_u, layers)

            dis, disw, ent, entw = entropy_coverage(neurons_states, test_state, layers-1)  

            # print(dis)
            # print(cov)

            for i in range(layers-1):
                cov_dis[i] += dis[i]   
                cov_add_dis[i].append(dis[i])  

            for i in range(layers-1):
                cov_disw[i] += disw[i]   
                cov_add_disw[i].append(disw[i])  
            for i in range(layers-1):
                cov_ent[i] += ent[i]   
                cov_add_ent[i].append(ent[i])  

            for i in range(layers-1):
                cov_entw[i] += entw[i]   
                cov_add_entw[i].append(entw[i])  

        cov_list_dis[e] = cov_dis   
        cov_list_disw[e] = cov_disw 
        cov_list_ent[e] = cov_ent  
        cov_list_entw[e] = cov_entw  

        # for me in methods:
        #     # print(me)
        #     attack_name = '{}_{}'.format(args_attack, error_rate)
        #     root_path = "./{}/".format(fid, error_rate)
        #     coverage = np.load(root_path + "{}_{}_{}.npy".format(me, attack_name, map_dataset))
        #     # print(list(coverage.tolist().values())[0])
        #     if me in ['NC', 'KMSC', 'MLC', 'MLSC']:
        #         dictall["{}".format(me)].append("{}".format(list(coverage.tolist().values())[0]))
        #     if me == 'lsa':
        #         dictall['LSC'].append("{}".format(list(coverage.tolist().values())[0]))
        #     if me == 'dsa':
        #         dictall['DSC'].append("{}".format(list(coverage.tolist().values())[0]))
        #     if me == 'nma':
        #         dictall['ANPC'].append("{}".format(list(coverage.tolist().values())[0]))
        #     if me == 'intra':
        #         dictall['SNPC'].append("{}".format(list(coverage.tolist().values())[0]))

            # cov_list_other[e] = list(coverage.tolist().values())[0]

        e = e + 1
        # print("Dis:", sum(cov_dis))
        # print("Disw:", sum(cov_disw))
        # print("Ent:", sum(cov_ent))
        # print("Entw:", sum(cov_entw))







