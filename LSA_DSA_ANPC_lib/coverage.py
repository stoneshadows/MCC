import os
import numpy as np
import copy
import heapq
# from utils.data_manger import *
# from attacks.attack_util import *
# from models.lenet import LeNet4
from collections import defaultdict
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
from deephunter.models.alexnet import AlexNet

os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cal_neu_cov(layer_output, mfr):
    k_sec = 1000
    # layer_output_list = []
    # for i in range(len(layer_output)):
    #     neuron_col = []
    #     for vector in layer_output[i]:
    #         neuron_col.append(np.transpose(vector))
    #     temp = np.transpose(neuron_col)
    #     layer_output_list.extend(temp)
    layer_output_list = []
    for i in range(len(layer_output)):
        layer_output[i] = np.array(layer_output[i]).reshape(np.array(layer_output[i]).shape[0], -1)
        temp = np.transpose(layer_output[i]).tolist()
        layer_output_list.extend(temp)
    MultiSecNeuCovNum, UpConNeuNum, LowConNeuNum = 0, 0, 0
    for n in range(len(layer_output_list)):
        n_msnc, n_ucn, n_lcn = 0, 0, 0
        # generate k-multisection zoom
        n_max, n_min = mfr[n][0], mfr[n][1]
        n_zoom = np.linspace(n_min, n_max, k_sec + 1)
        n_index = []
        # get the output of neuron n
        neuron_output = copy.deepcopy(layer_output_list[n])
        neuron_output = list(set(neuron_output))
        neuron_output.sort()
        # judge the value is belong which sub-zoom
        zoom_index = 0
        for value in neuron_output:
            if value < n_min:
                n_lcn = n_lcn + 1
            elif value > n_max:
                n_ucn = n_ucn + 1
            else:
                while zoom_index < len(n_zoom):
                    if value <= n_zoom[zoom_index]:
                        n_index.append(zoom_index)
                        break
                    zoom_index = zoom_index + 1
        n_msnc = len(set(n_index))
        MultiSecNeuCovNum = MultiSecNeuCovNum + n_msnc
        UpConNeuNum = UpConNeuNum + n_ucn
        LowConNeuNum = LowConNeuNum + n_lcn
    MultiSecNeuCov = MultiSecNeuCovNum / (k_sec * len(layer_output_list))
    NeuBoundCov = (UpConNeuNum + LowConNeuNum) / (2 * len(layer_output_list))
    StrNeuActCov = UpConNeuNum / len(layer_output_list)

    return MultiSecNeuCov, NeuBoundCov, StrNeuActCov

def cal_lay_cov(layer_output, mfr):
    top_k = 3
    NeuNum = len(mfr)
    print(len(mfr))
    TopKNeuNum = 0
    layer_output_list = []
    for i in range(len(layer_output)):
        neuron_col = []
        for vector in layer_output[i]:
            neuron_col.append(np.transpose(vector))
        temp = np.transpose(neuron_col)
        layer_output_list.extend(temp)
    # layer_output_list = []
    # for i in range(len(layer_output)):
    #     layer_output[i] = np.array(layer_output[i]).reshape(np.array(layer_output[i]).shape[0], -1)
    #     temp = np.transpose(layer_output[i]).tolist()
    #     layer_output_list.extend(temp)
    # 进行相关数据的计算
    top_k_neuron_layers = []
    for layer_index in range(len(layer_output_list)):
        top_k_neuron_layer = []
        for sample_index in range(len(layer_output_list[layer_index])):
            top_k_neuron_layer_sample = list(
                map(layer_output_list[layer_index][sample_index].index,
                    heapq.nlargest(top_k, layer_output_list[layer_index][sample_index])))
            top_k_neuron_layer.append(top_k_neuron_layer_sample)
        top_k_neuron_layers.append(top_k_neuron_layer)
    # topk数据保存方式：不同层-不同样本-对应的topk神经元位置
    # 计算TopKNeuNum
    for layer_index in range(len(top_k_neuron_layers)):
        temp = top_k_neuron_layers[layer_index]
        temp = np.array(temp).reshape(-1)
        TopKNeuNum = TopKNeuNum + len(set(temp))
    # 计算TKNPat
    TNKPatList = []
    for sample_index in range(len(layer_output_list[0])):
        sample_topk_pattern = []
        for layer_index in range(len(top_k_neuron_layers)):
            temp = top_k_neuron_layers[layer_index][sample_index]
            sample_topk_pattern.extend(temp)
        TNKPatList.append(hash(str(sample_topk_pattern)))

    TKNCov = TopKNeuNum / NeuNum
    TNKPat = len(set(TNKPatList))
    return TKNCov, TNKPat


mfr = np.load(f"./train_mfr.npy")
KMNC,NBC,SNAC = [list() for x in range(3)]

dataset = torchvision.datasets.SVHN(root='../deephunter/models/data/',
                                    transform=transforms.ToTensor(), download=True)

data_loader = Data.DataLoader(dataset=dataset, batch_size=32, shuffle=False)
model = AlexNet()
model.load_state_dict(torch.load("../data/trained_models/alexnet_lr0.0001_39.pkl"))
# model = models.alexnet(pretrained=True) # sy
model = model.cuda()
model.eval()

layers = 5
layer_outs = [[] for x in range(layers)]

for i, (data, target) in enumerate(data_loader):
    data, target = data.to(device), target.to(device)
    layer_out = model(data)

    for l in range(layers):
    ##训练集所有数据对应的每一层的神经元，转化成一行
        layer_outs[l].extend(layer_out[l].detach().reshape(1, -1))


print("KMNC=",cal_neu_cov(layer_outs, mfr)[0])
print("NBC=",cal_neu_cov(layer_outs, mfr)[1])
print("SNAC=",cal_neu_cov(layer_outs, mfr)[2])
print("TKNC=",cal_lay_cov(layer_outs, mfr)[0])
print("TKNP=",cal_lay_cov(layer_outs, mfr)[1])

# np.save(f"./normal/KMNC.npy", KMNC)
# np.save(f"./normal/NBC.npy", NBC)
# np.save(f"./normal/SNAC.npy", SNAC)
