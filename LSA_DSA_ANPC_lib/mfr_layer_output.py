import numpy as np
# from utils.data_manger import *
# from attacks.attack_util import *
# from models.lenet import LeNet4
from collections import defaultdict
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = 'cpu'
###   sy读取原样本数据
file_path = '../artifacts_eval/adv_samples/mnist/fgsm/2020-11-30 16-05-02/'
data = torchvision.datasets.MNIST(root=file_path, download=True, train= True,
                                  transform=torchvision.transforms.Compose(
                                     [torchvision.transforms.ToTensor(),normalize_mnist]))

# #对抗样本
# data = MyDataset(root=file_path, transform=transforms.Compose([
#     transforms.ToTensor(),normalize_mnist]), show_file_name=True, img_mode=None)

seed_model = LeNet4()
seed_model.load_state_dict(torch.load('../build-in-resource/pretrained-model/mnist/lenet.pkl'))
seed_model.eval()

loader = DataLoader(dataset=data)
model = seed_model.to(device)
# for index, l in model.named_parameters():
#     print(l)
for index, l in model._modules.items():
    print(index)
label = []
l0,l1,l2,l3,l4 = [list() for x in range(5)]

for data_tuple in loader:
    if len(data_tuple) == 2:
        data, target = data_tuple
    elif len(data_tuple) == 3:
        data, target, adv_label = data_tuple
    elif len(data_tuple) == 4:
        data, target, adv_label, file_name = data_tuple

    data, target = data.to(device), target.to(device)
    layer_out = model(data)

    ##训练集所有数据对应的每一层的神经元，转化成一行
    l0.extend(layer_out[0].detach().numpy().reshape(1,-1))
    # l1.extend(layer_out[1].detach().numpy().flatten())
    l1.extend(layer_out[1].detach().numpy().reshape(1,-1))
    l2.extend(layer_out[2].detach().numpy().reshape(1,-1))
    l3.extend(layer_out[3].detach().numpy().reshape(1,-1))
    l4.extend(layer_out[4].detach().numpy().reshape(1,-1))
    label.extend(target.tolist())  # 所有标签列表


layer_output = [_ for _ in (l0,l1,l2,l3,l4)]

# np.save(f"./normal/layer_output.npy", layer_output)
# np.save(f"./normal/train_label.npy", label)
layer_output_list = []
for i in range(len(layer_output)):
    layer_output_list = []
    for i in range(len(layer_output)):
        layer_output[i] = np.array(layer_output[i]).reshape(np.array(layer_output[i]).shape[0], -1)
        temp = np.transpose(layer_output[i]).tolist()
        layer_output_list.extend(temp)

# 对相应位置数据的major function region进行求取
mfr = []
for i in range(len(layer_output_list)):  # 逐个访问神经元
    # print(i)
    maxvalue = max(layer_output_list[i])
    minvalue = min(layer_output_list[i])
    mfr.append([maxvalue, minvalue])
# 对mfr(major function region)进行保存
mfr = np.array(mfr)
np.save(f"./normal/train_mfr.npy", mfr)
# np.save(f"./normal/layer_output_list.npy", layer_output_list)




# for index,l in model.named_parameters():
#     print(np.max(l))

# samples_filter(seed_model, DataLoader(dataset=data), 'legitimate {} >>'.format('mnist'), size=-1,
#                show_progress=False, device= 'cpu', is_verbose=True)