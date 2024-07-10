import pickle
import argparse
import numpy as np
import math
import torch
from collections import Counter
from LSA_DSA_ANPC_lib.new_sa_torch import fetch_dsa, fetch_lsa, get_sc, fetch_newMetric


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 计算相邻层的转移概率矩阵
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
            if matrix_train[i][j] == 0:
                continue
            elif matrix_train[i][j] != 0 and matrix_test[i][j] == 0:
                dis = dis - 1
            else:
                dis = dis - (matrix_train[i][j] * math.log(matrix_test[i][j]))
    return dis



def test_states(paths, neurons_states, samples_class, picked_units, num_layers):
    test_state_dict = [{} for _ in range(num_layers)]
    for layer in range(num_layers):
        keys_list = list(neurons_states[layer].keys())
        test_state_dict[layer] = dict([(key, []) for key in keys_list])

    test_num = [[] for _ in range(num_layers)]

    for index in samples_class:  
        n_state = [[] for _ in range(num_layers)]
        for layer in range(num_layers):
            n = paths[index][layer]  

            for n_i in n:
                if n_i in picked_units[layer]:  
                    n_state[layer].append(n_i)  

            n_state[layer].sort()
            layer_state = neurons_states[layer]
            keys_list = list(layer_state.keys())

            if str(n_state[layer]) in keys_list:
                test_num[layer].append(index)
                test_state_dict[layer][str(n_state[layer])].append(index)  
    return test_state_dict


def entropy_coverage(neurons_states, test_states, layer):
    dis1 = []
    # trans_list_train = []
    # trans_list_test = []

    numall = [0 for _ in range(layer)]
    for i in range(layer):
        if i != layer:
            dic1 = neurons_states[i]
            numall[i] = len(dic1)
            dic2 = neurons_states[i + 1]
            ma1 = matrix(dic1, dic2)
            # trans_list_train.append(matrix(dic1, dic2))

            dic11 = test_states[i]
            dic22 = test_states[i + 1]
            ma2 = matrix(dic11, dic22)
            # trans_list_test.append(matrix(dic11, dic22))

            c_dis = C_dis(ma1, ma2)
            dis1.append(c_dis)

    return sum(dis1)

def selectsample(samples_class, batch, iterate, max_iter):
    arr = np.random.permutation(samples_class)
    arr_size = int(len(samples_class) * 0.6)
    max_index0 = arr[0:arr_size]
    min_index0 = arr[0:arr_size]
    acc_list1 = []
    acc_list2 = []
    cov_random = []
    cov_select = []

    for i in range(iterate):  
        arr = np.random.permutation(samples_class)

        temp_cov = []
        index_list = []


        x_test = dataset.data[max_index0] 
        # y_test = dataset.labels[max_index0]
        y_test = np.array(dataset.targets)[max_index0]

        # x_test = x_test.transpose(0, 2, 3, 1)
        # x_test = x_test.astype("float32")
        if len(x_test.shape) <= 3:
            x_test = np.expand_dims(x_test, axis=1)
        if x_test.shape[1] not in [1, 3]:
            x_test = x_test.transpose(0, 3, 1, 2)
        _test = torch.utils.data.TensorDataset(torch.FloatTensor(x_test), torch.LongTensor(y_test))
        # cov = neuron_coverage(model, _test)[-1]
        # cov = k_multi_section_coverage(model, x_train, _test)[-1]
        # cov = multi_layer_coverage(model,  _test)[-1]
        cov = multi_layer_section_coverage(model, x_train, _test)[-1]

        # target_name =  "_test" + fid
        #
        # target_lsa = fetch_lsa(model, x_train, x_test, target_name,
        #                        [layer_names[-1]], args, cluster_paths, num_layer - 1, path=args.path)
        #
        # target_cov, buckets = get_sc(np.amin(target_lsa), np.amax(target_lsa), n_bucket, target_lsa)  # 得到覆盖率
        # # test_lsa_score = np.mean(test_lsa)
        #
        # target_lsa = fetch_lsa(model, x_train, x_test, target_name,
        #                        [layer_names[-1]], args, cluster_paths, num_layer - 1, path=args.path)
        #
        # # print (target_lsa)
        # # target_cov, buckets = get_sc(np.amin(target_lsa), 2000, n_bucket, target_lsa)  # 得到覆盖率
        # target_cov, buckets = get_sc(np.amin(target_lsa), np.amax(target_lsa), n_bucket, target_lsa)
        #
        # print("cov:", cov)
        # test_state_max = test_states(paths, neurons_states, max_index0, picked_units, layers )
        # cov = entropy_coverage(neurons_states, test_state_max, layers-1)

        for j in range(max_iter): 
            arr = np.random.permutation(samples_class)
            start = int(np.random.uniform(0, len(samples_class) - batch))
            temp_index = np.append(max_index0, arr[start:start + batch])  
            index_list.append(arr[start:start + batch])

            # test_state1 = test_states(paths, neurons_states, temp_index, picked_units, layers)
            # new_coverage = entropy_coverage(neurons_states, test_state1, layers-1)
            x_test = dataset.data[temp_index] 
            # y_test = dataset.labels[temp_index]
            y_test = np.array(dataset.targets)[temp_index]

            # x_test = x_test.transpose(0, 2, 3, 1)
            # x_test = x_test.astype("float32")
            if len(x_test.shape) <= 3:
                x_test = np.expand_dims(x_test, axis=1)
            if x_test.shape[1] not in [1, 3]:
                x_test = x_test.transpose(0, 3, 1, 2)
            _test = torch.utils.data.TensorDataset(torch.FloatTensor(x_test), torch.LongTensor(y_test))

            # new_coverage = neuron_coverage(model, _test)[-1]
            # new_coverage = k_multi_section_coverage(model, x_train, _test)[-1]
            # new_coverage = multi_layer_coverage(model,  _test)[-1]
            new_coverage = multi_layer_section_coverage(model, x_train, _test)[-1]
            print("new_cover:", new_coverage)

            temp_cov.append(new_coverage)

        max_coverage = np.max(temp_cov)
        cov_index = np.argmax(temp_cov)  
        max_index = index_list[cov_index]

        if (max_coverage <= cov):
            start = int(np.random.uniform(0, len(samples_class) - batch))
            max_index = arr[start:start + batch] 
        max_index0 = np.append(max_index0, max_index)

        # test_state_max = test_states(paths, neurons_states, max_index0, picked_units, layers )
        # cov1 = entropy_coverage(neurons_states, test_state_max, layers-1)

        x_test = dataset.data[max_index0]  
        # y_test = dataset.labels[max_index0]
        y_test = np.array(dataset.targets)[max_index0]

        # x_test = x_test.transpose(0, 2, 3, 1)
        # x_test = x_test.astype("float32")
        if len(x_test.shape) <= 3:
            x_test = np.expand_dims(x_test, axis=1)
        if x_test.shape[1] not in [1, 3]:
            x_test = x_test.transpose(0, 3, 1, 2)
        _test = torch.utils.data.TensorDataset(torch.FloatTensor(x_test), torch.LongTensor(y_test))
        # cov1 = neuron_coverage(model, _test)[-1]
        # cov1 = k_multi_section_coverage(model, x_train, _test)[-1]
        # cov1 = multi_layer_coverage(model,  _test)[-1]
        cov1 = multi_layer_section_coverage(model, x_train, _test)[-1]
        print("cov1:", cov1)

        cov_select.append(cov1)
        # cov_random.append(cov2)
        print("current select coverage is {!s}".format(cov1))
        # print("current random coverage is {!s}".format(cov2))

        fid = 'diversity4.21'
        daset = 'cifar10vgg'
        root_paths = "./{}/{}/".format(fid, daset)
        cluss = np.load(root_paths + 'fault_cluster2.npy'.format(fid))

        n_max = 0
        n_min = 0
        num_c = []

        for ma in set(max_index0):
            if ma not in right_samples_class:
                n_max += 1
                # 4.23
                for i, clu in enumerate(cluss):
                    if ma in clu:
                        num_c.append(i)
        acc2 = len(list(set(num_c)))
        acc1 = n_max

        print(acc1)
        print(acc2)
        acc_list1.append(acc1)
        acc_list2.append(acc2)

        # for mi in set(min_index0):
        #     if mi in right_samples_class:
        #         n_min += 1
        # acc2 = n_min / len(set(min_index0))


        print("numuber of samples is {!s}, select acc is {!s}:"
              .format(len(max_index0), acc1))
    return acc_list1, acc_list2, cov_select

def experiments(right_samples_class, samples_class, batch, iterate, max_iter):
    true_acc = len(right_samples_class) / len(samples_class)
    print("The final acc is {!s}".format(true_acc))
    acc_list1, acc_list2,  cov1 = selectsample(samples_class, batch, iterate, max_iter)
    return acc_list1, acc_list2, cov1

def avg_cla(acc_classes):
    ini_matrix = np.zeros((len(acc_classes[0]), len(acc_classes[0][0])))
    for cla in range(classes):
        for i in range(len(acc_classes[cla])):
            for j in range(len(acc_classes[cla][i])):
                ini_matrix[i][j] += acc_classes[cla][i][j]

    for i in range(len(ini_matrix)):
        for j in range(len(ini_matrix[i])):
            ini_matrix[i][j] = ini_matrix[i][j]/classes
    return ini_matrix

# def nature_error(error_rate, right_class, wrong_class):
#     classes = 10
#     nat_class = [[] for _ in range(classes)]
#     nat_all = []
#     for cla in range(classes):
#         if error_rate == 1.:
#             nat_class[cla] = right_class[cla] + wrong_class[cla]
#         else:
#             nat_class[cla] = right_class[cla]
#     for c in range(classes):
#         for l in nat_class[c]:
#             nat_all.append(l)
#     return nat_class, nat_all

if __name__ == '__main__':
    import random
    import numpy as np
    import pickle
    import torch

    import torchvision
    import torchvision.transforms as transforms
    import torch.utils.data as Data
    from deephunter.models.alexnet import AlexNet
    from LSA_DSA_ANPC_lib.utils_data import get_dataset
    from neuron_op import neuron_coverage
    from neuron_op import k_multi_section_coverage
    from neuron_op import multi_layer_coverage
    from neuron_op import multi_layer_section_coverage
    from deephunter.models.covnet_mnist import ConvnetMnist
    from deephunter.models.covnet_cifar10 import ConvnetCifar
    from deephunter.models.vgg_cifar10 import VGG16

    classes = 10
    # root_path = "./cluster_paths/{}/".format(args.arc)

    feature_size = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    layers = len(feature_size)
    fid = "adv_cifar_vgg"
    map_dataset = 'cifar10'
    map_arg = 'vgg'


    # feature_size = [32, 32, 64, 64, 128, 128, 1024, 512]
    # layers = len(feature_size)
    # fid = "adv_cifar"
    # map_dataset = 'cifar10'
    # map_arg = 'convcifar10'

    # fid = 'adv_svhn'
    # map_arg = 'alexnet'
    # map_dataset = 'alexnet'
    #
    # feature_size = [64, 192, 384, 256, 256]

    # fid = 'adv_mnist'
    # map_arg = 'convmnist'
    # map_dataset = 'mnist'
    # feature_size = [64, 64, 128]


    layers = len(feature_size)

    data_file0 = open('./{}/all_right_class_index.pkl'.format(fid), 'rb')
    right_index = pickle.load(data_file0)
    data_file0.close()

    data_file1 = open('./{}/all_wrong_class_index.pkl'.format(fid), 'rb')
    wrong_index = pickle.load(data_file1)
    data_file1.close()

    # root_path = "./cluster_paths/{}/".format(args.arc)
    # data_path = "./cluster_paths_0/SVHN_alexnet_lrp_path_threshold0.8_train.pkl"
    # data_path = "./{}/cluster_paths_0/PGD_0_lrp_path_threshold0.8_test.pkl".format(fid)
    # with open(data_path, 'rb') as fr:
    #     paths = pickle.load(fr)
    ######

    # dataset = torchvision.datasets.MNIST(root='./deephunter/models/data/',
    #                                      transform=transforms.ToTensor(), train=False)
    # dataset = torchvision.datasets.SVHN(root='./deephunter/models/data/',
    #                                     transform=transforms.ToTensor(), download=True, split='test')

    dataset = torchvision.datasets.CIFAR10(root='./deephunter/models/data/',
                                        transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((128,128))]), download=True, train=False)

    x_train = get_dataset(map_dataset)

    from torchvision import models
    import torch.nn as nn
    model_save_path = "./data/trained_models/ResNet_CIFAR.pth"
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    model.load_state_dict(torch.load(model_save_path))

    # model = AlexNet()
    # model.load_state_dict(torch.load("./data/trained_models/alexnet.pkl"))

    # model = ConvnetMnist()
    # model.load_state_dict(torch.load("./data/trained_models/mnist.pth"))

    # model = ConvnetCifar()
    # model.load_state_dict(torch.load("./data/trained_models/cifar.pth")["net"])
    #
    # model = VGG16()
    # model.load_state_dict(torch.load("./data/trained_models/vgg.pkl"))
    # data_loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    # model = models.alexnet(pretrained=True) # sy
    model = model.cuda()
    model.eval()

    #########
    # ori_model, layer_names, num_layer = get_model(args.dataset, args.arch)
    # assert layer_names is not None and len(layer_names) > 0, f"expect layer_names[] >0, layer_names= {layer_names}"
    # ori_model = ori_model.cuda()
    # model = TorchModel(ori_model, layer_names=layer_names)
    # model.eval()
    # ori_model.eval()


    cover = []
    samples_set = []
    new_sample_set = []


    total = 0
    exper = 1
    acc1_classes = [[] for _ in range(classes)]
    acc11_classes = [[] for _ in range(classes)]
    acc2_classes = [[] for _ in range(classes)]
    cov1_classes = [[] for _ in range(classes)]
    cov2_classes = [[] for _ in range(classes)]

    for cla in range(classes):
    # for cla in [0]:
        print("class:", cla)

        samples_class = right_index[cla] + wrong_index[cla]
        num_picked_samples = len(samples_class)

        right_samples_class = right_index[cla]


        for k in range(exper):
            print("the {} exp".format(k))
            # acc1,  cov1 = experiments(right_samples_class, samples_class, batch=600, iterate=15, max_iter=5)
            acc1, acc11, cov1 = experiments(right_samples_class, samples_class, batch=30, iterate=30, max_iter=5)
            acc1_classes[cla].append(acc1)
            acc11_classes[cla].append(acc11)
            # acc2_classes[cla].append(acc2)
            cov1_classes[cla].append(cov1)
            # cov2_classes[cla].append(cov2)
    #
    accc1 = avg_cla(acc1_classes)
    acc11 = avg_cla(acc11_classes)
    # acc22 = avg_cla(acc2_classes)
    cov11 = avg_cla(cov1_classes)
    # cov22 = avg_cla(cov2_classes)
    #
    for k1 in range(exper):
        print(np.array(accc1[k1]))
        print(np.array(acc11[k1]))
        # np.savetxt('data/select{}_{}.csv'.format(k1, fid), np.array(acc11[k1]))
        # np.savetxt('data/select1{}_{}.csv'.format(k1, fid), np.array(acc11[k1]))
        # np.savetxt('data/4.23_mls1{}_resnet.csv'.format(k1), np.array(accc1[k1]))
        # np.savetxt('data/4.23_mls11{}_resnet.csv'.format(k1), np.array(acc11[k1]))
    #     np.savetxt('data/random{}_{}.csv'.format(k1, fid), np.array(acc22[k1]))
    #     np.savetxt('data/select_cov{}_{}.csv'.format(k1, fid), np.array(cov11[k1]))
    #     np.savetxt('data/select1cov{}_{}.csv'.format(k1, fid), np.array(cov11[k1]))
    #     np.savetxt('data/select_mlsc1cov{}_{}.csv'.format(k1, fid), np.array(cov11[k1]))
    #     np.savetxt('data/random_cov{}_{}.csv'.format(k1, fid), np.array(cov22[k1]))
    #     print("save done!!!")
