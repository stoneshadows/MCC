# python path_LRP.py --gpu 0 --arc vgg16_bn --threshold 0.7 --dataset imagenet --suffix imagenet --data_train
import numpy as np
import sys
sys.path.append(".")

from LSA_DSA_ANPC_lib.LRP_path.innvestigator import InnvestigateModel
from LSA_DSA_ANPC_lib.LRP_path.inverter_util import Flatten
from neuron_op import neuron_coverage, k_multi_section_coverage, multi_layer_section_coverage, multi_layer_coverage
from LSA_DSA_ANPC_lib.utils_data import get_dataset
from LSA_DSA_ANPC_lib.new_sa_torch import fetch_dsa, fetch_lsa, get_sc
from LSA_DSA_ANPC_lib.torch_modelas_keras import TorchModel
from LSA_DSA_ANPC_lib.neuron_coverage import Coverager
from LSA_DSA_ANPC_lib import utils_data as calc_sadl_utils_data
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
from scipy import stats

random.seed(1024)
np.random.seed(1024)

parser = argparse.ArgumentParser()
# parser.add_argument("--d", "-d", help="Dataset", type=str, default="svhn")
# parser.add_argument(
#     "--lsa", "-lsa", help="Likelihood-based Surprise Adequacy", action="store_true"
# )
parser.add_argument(
    "--dsa", "-dsa", help="Distance-based Surprise Adequacy", action="store_true"
)
parser.add_argument(
    "--nma", "-nma", help="NM Adequacy", action="store_true"
)
parser.add_argument(
    "--last_layer", action="store_true"
)
parser.add_argument(
    "--path", action="store_true"
)
parser.add_argument(
    "--rest", action="store_true"
)

parser.add_argument(
    "--save_path", "-save_path", help="Save path", type=str, default="./feature_maps/vgg/"
)
parser.add_argument(
    "--batch_size", "-batch_size", help="Batch size", type=int, default=128
)
parser.add_argument(
    "--var_threshold",
    "-var_threshold",
    help="Variance threshold",
    type=int,
    default=1e-5,
)
parser.add_argument(
    "--upper_bound", "-upper_bound", help="Upper bound", type=int, default=2000
)
parser.add_argument(
    "--n_bucket",
    "-n_bucket",
    help="The number of buckets for coverage",
    type=int,
    default=1000,
)
parser.add_argument(
    "--num_classes",
    "-num_classes",
    help="The number of classes",
    type=int,
    default=10,
)
parser.add_argument(
    "--is_classification",
    "-is_classification",
    help="Is classification task",
    type=bool,
    default=True,
)
parser.add_argument('--gpu', type=str, default="0")
parser.add_argument('--arch', type=str, default="alexnet")
parser.add_argument('--dataset', type=str, default="svhn")

args = parser.parse_args()

from LSA_DSA_ANPC_lib.utils_data import get_model, get_dataset, get_cluster_para

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


def generate_class(testset, m):
    classes = [[] for i in range(10)]
    for idx, (image, label) in enumerate(testset):
        classes[label].append(idx)

    test_set1 = []
    for ccla in classes:
        s_data = random.sample(ccla, m)
        test_set1.extend(s_data)

    return test_set1


fid = 'results'
daset = 'svhn'
arc = 'alexnet'
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


root_path = "./{}/{}/".format(fid, daset)


allspear = [[] for _ in range(4)]
allpvalue = [[] for _ in range(4)]
for ze, si in enumerate([100,200,300,400]):
    nccov, kmsccov, mlccov, mlsccov, lsacov, dsacov, anpccov, snpccov = [], [], [], [], [], [], [], []
    dis1, disw1, ent1, entw1 = [], [], [], []
    num_f = []
    for _ in range(20):
        cla_set = np.random.choice(len(testset), size=si, replace=True)
        # cla_set = generate_class(testset, int(si/10))
        attack_name = 'RQ2_{}'.format(si)
        samples_class = [[] for c in range(10)]
        right_samples_class = [[] for c in range(10)]
        paths_class = [[] for c in range(10)]

        selected_data = [testset[i] for i in cla_set]

        data_loader = Data.DataLoader(dataset=selected_data, batch_size=batch_size, shuffle=False)

        cluss = np.load(root_path + 'fault_cluster.npy')

        num_c = []
        for c in cla_set:
            for i, clu in enumerate(cluss):
                if c in clu:
                    num_c.append(i)
 
        num_clu = len(list(set(num_c)))

        num_f.append(num_clu)
        print(num_clu)


##########  # other coverage

        x_train = get_dataset(daset)

        x_test = testset.data[cla_set]
        y_test = np.array(testset.targets)[cla_set]

        # x_test = testset.data[cla_set]
        # y_test = testset.labels[cla_set]

        # x_test = x_test.transpose(0, 2, 3, 1)
        # x_test = x_test.astype("float32")

        if len(x_test.shape) <= 3:
            x_test = np.expand_dims(x_test, axis=1)
        if x_test.shape[1] not in [1, 3]:
            x_test = x_test.transpose(0, 3, 1, 2)
        # x_test = torch.utils.data.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
        x_test = torch.utils.data.TensorDataset(torch.FloatTensor(x_test), torch.LongTensor(y_test))
        dataloader = Data.DataLoader(dataset=x_test, batch_size=32, shuffle=False)


        coverage1 = neuron_coverage(model, x_test)
        nccov.append(coverage1[-1])
        print(coverage1[-1])

        coverage2 = k_multi_section_coverage(model, x_train, x_test)
        kmsccov.append(coverage2[-1])
        print(coverage2[-1])

        coverage4 = multi_layer_coverage(model, x_test)
        mlccov.append(coverage4[-1])
        print(coverage4[-1])

        coverage5 = multi_layer_section_coverage(model, x_train, x_test)
        mlsccov.append(coverage5[-1])
        print(coverage5[-1])



    #### lsa and dsa coverage

        ori_model1, layer_names, num_layer = get_model(args.dataset, args.arch)
        assert layer_names is not None and len(layer_names) > 0, f"expect layer_names[] >0, layer_names= {layer_names}"
        ori_model1 = ori_model1.cuda()
        model1 = TorchModel(ori_model1, layer_names=layer_names)
        model1.eval()

        n_bucket = args.n_bucket

        sample_threshold, cluster_threshold = get_cluster_para(args.dataset, args.arch)

        cluster_paths = [[] for _ in range(10)]

        for cla in range(10):
            picked_samples_fname = path_train + "threshold{}_class{}_paths.pkl".format(threshold, cla)

            assert os.path.isfile(picked_samples_fname), f"expect the file {picked_samples_fname} "
            with open(picked_samples_fname, "rb") as f:
                try:
                    unpickler = pickle.Unpickler(f)
                    path = unpickler.load()
                    cluster_paths[cla].append(path)
                except Exception as ex:
                    print(picked_samples_fname, ex)
                    pass

        target_name = attack_name + "_test" + fid

        target_lsa = fetch_lsa(model1, x_train, x_test, target_name,
                               [layer_names[-1]], args, cluster_paths, num_layer - 1, path=args.path)

        target_cov1, _ = get_sc(np.amin(target_lsa), np.amax(target_lsa), args.n_bucket, target_lsa)  # 得到覆盖率
        print('lsa:', target_cov1)
        lsacov.append(target_cov1)

        ##
        target_dsa, _, a_dists, b_dists = fetch_dsa(model1, x_train, x_test, target_name,
                                                    [layer_names[-1]], args, cluster_paths, num_layer - 1,
                                                    path=args.path)
        target_cov, _ = get_sc(np.amin(target_dsa), np.amax(target_dsa), args.n_bucket, target_dsa)

        dsacov.append(target_cov)
        print("dsa:", target_cov)



        ###### cal_SNPC
        from deephunter.models import get_net
        covered_10 = set()
        covered_100 = set()
        total_100 = total_10 = 0
        bucket_m = 200
        fetch_func = lambda x: x[0]
        nn_model = get_net(name=args.arch)
        for step, x in enumerate(dataloader):
            one_data = fetch_func(x)
            x = one_data.to(device)

            cover = Coverager(nn_model, args.arch, cluster_threshold, num_classes=10)
            # print ("model_name",model_name,"cluster_threshold",cluster_threshold,"num_classes",num_classes,"clust",cluster_num)

            start_time1 = time.time()
            covered1, total1 = cover.Intra_NPC(x, bucket_m, sample_threshold,  simi_soft=False,
                                               arc=args.arch)
            start_time2 = time.time()
            covered2, total2 = cover.Layer_Intra_NPC(x, bucket_m, sample_threshold,  simi_soft=False,
                                                     useOldPaths_X=True, arc=args.arch)
            start_time3 = time.time()

            total_10 += total1
            total_100 += total2
            covered_10 = covered_10 | covered1
            covered_100 = covered_100 | covered2


        intra = round(len(covered_10) / total1, 5)
        layer_intra = round(len(covered_100) / total2, 5)

        print('ANPC:', intra)
        print('SNPC:', layer_intra)
        anpccov.append(intra)
        snpccov.append(layer_intra)



###########################  my coverage

        inn_model = InnvestigateModel(model, lrp_exponent=2, method="b-rule", beta=.5)

        s_time = time.time()

        for i, (data, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
            data, target = data.cuda(), target.cuda()
            batch_size = int(data.size()[0])

            # if i in attacklist:
            model_prediction, _, true_relevance = inn_model.innvestigate(in_tensor=data)
            true_relevance = true_relevance[:]

            # if i == min(attacklist):
            if i == 0:  
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

        #####
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
                # print(sta)

            picked_units = [[] for _ in range(num_layers)]
            for layer in range(num_layers):
                for t, s in enumerate(sta[layer]):
                    if s < round(len(cla_set) * threshold):  
                        picked_units[layer] = sensUnits[layer][:t + 1]
                        if t * 2 < len(sensUnits[layer]):  
                            sec_picked_units[layer] = sensUnits[layer][t: t * 2]  # [1:2], [2:4], [3:6]
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

    
        test_state = test_states(paths, neurons_states, cla_set, picked_units, num_layers)
        dis, disw, ent, entw = entropy_coverage(neurons_states, test_state, num_layers-1)  


        dis1.append(sum(dis))
        disw1.append(sum(disw))
        ent1.append(sum(ent))
        entw1.append(sum(entw))

        # print(sum(dis))
        # print(sum(ent))


    for c, cov_list in enumerate([nccov, kmsccov, mlccov, mlsccov, lsacov, dsacov, anpccov, snpccov,ent1]):
        np.save(root_path + 'cov_{}_{}.npy'.format(ze,c), cov_list)
        np.save(root_path + 'fault_{}_{}.npy'.format(ze, c), num_f)
        spearman, p_value = stats.spearmanr(cov_list, num_f)
        allspear[ze].append(spearman)
        allpvalue[ze].append(p_value)

    print('corr_of_size_{}:'.format(si), spearman, p_value)
    print(allspear)
    print(allpvalue)
    np.save(root_path + 'spearmancorr.npy', allspear)
    np.save(root_path + 'p_values.npy', allpvalue)



