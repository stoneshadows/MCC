import torch
import torchvision
import os
import torchvision.transforms as transforms
from collections import Counter
import torch.utils.data as Data

import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if args.arc == "vgg" or args.arc == "vgg16_bn":
#     feature_size = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
# elif args.arc == "resnet18":
#     feature_size = [64, 64, 64, 128, 128, 256, 256, 512, 512]
# elif args.arc == "convmnist":
#     feature_size = [64, 64, 128]
# elif args.arc == "convcifar10":
#     feature_size = [32, 32, 64, 64, 128, 128, 1024, 512]
# elif args.arc == "alexnet":
#     feature_size = [64, 192, 384, 256, 256]


def adv_error(error_rate, right_class, wrong_class):
    classes = 10
    adv_class = [[] for _ in range(classes)]
    for cla in range(classes):
        adv_class[cla] = right_class[cla] + wrong_class[cla]
    return adv_class

import random
import Adversarial_Attack as AA


threshold = 0.8
batch_size = 32



classes = 10
# fid = 'adv_svhn1'  
# map_arg = 'alexnet'
# map_dataset = 'svhn'
# feature_size = [64, 192, 384, 256, 256]

# fid = "adv_vgg"
# map_dataset = 'cifar10'
# feature_size = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]

# fid = "adv_cifar_vgg"
# map_dataset = 'cifar10'
# feature_size = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]

fid = 'adv_mnist1'
map_dataset = 'mnist'
feature_size = [64, 64, 128]


# fid = "adv_cifar1"
# map_dataset = 'cifar10'
# feature_size = [32, 32, 64, 64, 128, 128, 1024, 512]

args_attack = 'PGD'
# error_rate_list = [.01, .02, .04, .06, .08, .1]
# error_rate_list = [0.01, .1, .2, .3, .4, .5, .6, .7, .8, .9]
# error_rate_list = [100, 200, 400, 600, 800, 1000]
error_rate_list = [100, 1000, 2000]
data_file = open('./{}/all_right_class_index.pkl'.format(fid), 'rb')
right_class = pickle.load(data_file)
data_file.close()


data_path = "./adv_cifar_vgg/cluster_paths_0/PGD_0_lrp_path_threshold0.8_test.pkl"
with open(data_path, 'rb') as fr:
    paths = pickle.load(fr)  

for error_rate in error_rate_list:
    print(error_rate)
    attack_name = '{}_{}'.format(args_attack, error_rate)


    root_path0 = "./{}/cluster_paths_{}/".format(fid, error_rate)
    data_path = root_path0 + "{}_lrp_path_threshold0.8_test.pkl".format(attack_name)
    with open(data_path, 'rb') as fr1:
        adv_paths = pickle.load(fr1)

    data_path1 = root_path0 + '{}_attacklist_{}_{}.pkl'.format(map_dataset, args_attack, error_rate)
    with open(data_path1, 'rb') as fr2:
        attacklist = pickle.load(fr2)

    data_path2 = root_path0 + '{}_attackclass_{}_{}.pkl'.format(map_dataset, args_attack, error_rate)
    with open(data_path2, 'rb') as fr3:
        attacklistclass = pickle.load(fr3)

    root_path = "./{}/cluster_paths_{}/{}/".format(fid, error_rate, attack_name)
    if os.path.exists(root_path) == False:
        os.makedirs(root_path)

    for cla in range(classes):  

        num_layers = len(feature_size)

        # picked_samples_fname = root_path + "samples_class{}.pkl".format(cla)
        # output = open(picked_samples_fname, 'wb')
        # pickle.dump(right_class[cla], output)            

        num_picked_samples = len(right_class[cla])

        neurons = [[] for _ in range(num_layers)]
        neurons_state = [[] for _ in range(num_layers)]
        sensUnits = [[] for _ in range(num_layers)]
        sec_picked_units = [[] for _ in range(num_layers)]
        sta = [[] for _ in range(num_layers)]

        a_c = root_path + "adv_class{}.pkl".format(cla)
        out = open(a_c, 'wb')
        adv_class = adv_error(error_rate, right_class, attacklistclass)
        print(len(adv_class[cla]))
        pickle.dump(adv_class[cla], out)


        for index in adv_class[cla]:
            for layer in range(num_layers):
                if index in attacklistclass[cla]:
                    i = attacklist.index(index)
                    n = adv_paths[i][layer]   
                    # n = paths[index][layer]
                else:
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
                if s < round(num_picked_samples * threshold):  
                    picked_units[layer] = sensUnits[layer][:t+1]


                    if t * 2 < len(sensUnits[layer]):  
                        sec_picked_units[layer] = sensUnits[layer][t: t * 2]  
                    else:
                        sec_picked_units[layer] = sensUnits[layer][t:]
                    break

                if t == len(sta[layer])-1:
                    print("warning")
                    picked_units[layer] = sensUnits[layer]

        path_fname = root_path + "threshold{}_class{}_paths.pkl".format(threshold, cla)
        output = open(path_fname, 'wb')
        pickle.dump(picked_units, output)












