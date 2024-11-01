import os
import torch
import pickle
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
from deephunter.models.alexnet import AlexNet
from deephunter.models.covnet_mnist import ConvnetMnist
from deephunter.models.covnet_cifar10 import ConvnetCifar
from deephunter.models.vgg_cifar10 import VGG16
import random
import numpy as np
import Adversarial_Attack as AA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dataset = torchvision.datasets.SVHN(root='./deephunter/models/data/',
#                                transform = transforms.ToTensor(), split='test')
#
# data_loader = Data.DataLoader(dataset=dataset, batch_size=32, shuffle=False)
#
# model = AlexNet()
# model.load_state_dict(torch.load("./data/trained_models/alexnet_lr0.0001_39.pkl"))
#
#
# dataset = torchvision.datasets.MNIST(root='./deephunter/models/data/', train=False,
#                                      transform=transforms.ToTensor(), download=True)
# #
# #
# model = ConvnetMnist()
# model.load_state_dict(torch.load("./data/trained_models/mnist_mixup_acc_99.28_ckpt.pth")["net"])

dataset = torchvision.datasets.CIFAR10(root='./deephunter/models/data/', transform=transforms.ToTensor(), train=False,
                                       download=True)
#
# model = ConvnetCifar()
# model.load_state_dict(torch.load("./data/trained_models/cifar_mixup_acc_90.36_ckpt.pth")["net"])

model = VGG16()
model.load_state_dict(torch.load("./data/trained_models/vgg_seed32_dropout.pkl"))

model = model.cuda()
model.eval()

classes = 10
right_i = []
right_class = [[] for c in range(classes)]
wrong_i = []
wrong_class = [[] for c in range(classes)]
adv_loader = Data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)  # batch_size必须是1
for num, (data, target) in enumerate(adv_loader):
    data, target = data.to(device), target.to(device)
    ori_data = data.data
    output = model(ori_data)
    _, init_pred = output.max(1)

    if init_pred == target:
        right_class[target.item()].append(num)
        right_i.append(num)
    else:
        wrong_class[target.item()].append(num)
        wrong_i.append(num)

file = 'adv_vgg'
# file = 'adv_cifar1'

if os.path.exists('./{}/'.format(file)) == False:
    os.makedirs('./{}/'.format(file))
pickle.dump(right_class, open('./{}/all_right_class_index.pkl'.format(file), 'wb'))
pickle.dump(right_i, open('./{}/all_right_i.pkl'.format(file), 'wb'))
print("correct index have saved!!!")

pickle.dump(wrong_class, open('./{}/all_wrong_class_index.pkl'.format(file), 'wb'))
pickle.dump(wrong_i, open('./{}/all_wrong_i.pkl'.format(file), 'wb'))
print("error index have saved!!!")


if __name__=="__main__":
    from deephunter.models.alexnet import AlexNet
    from deephunter.models.covnet_mnist import ConvnetMnist
    from deephunter.models.covnet_cifar10 import ConvnetCifar
    from deephunter.models.vgg_cifar10 import VGG16
    # import numpy as np
    import pickle
    # dataset = torchvision.datasets.SVHN(root='./deephunter/models/data/',
    #                                transform = transforms.ToTensor(), split='test')
    # data_loader = Data.DataLoader(dataset=dataset, batch_size=32, shuffle=False)
    # model = AlexNet()
    # model.load_state_dict(torch.load("./data/trained_models/alexnet_lr0.0001_39.pkl"))

    # dataset = torchvision.datasets.MNIST(root='./deephunter/models/data/', train=False,
    #                                      transform=transforms.ToTensor(), download=True)
    # data_loader = Data.DataLoader(dataset=dataset, batch_size=32, shuffle=False)


    # model = ConvnetMnist()
    # model.load_state_dict(torch.load("./data/trained_models/mnist_mixup_acc_99.28_ckpt.pth")["net"])


    dataset = torchvision.datasets.CIFAR10(root='./deephunter/models/data/', transform=transforms.ToTensor(), train = False,
                                download=True)

    # model = ConvnetCifar()
    # model.load_state_dict(torch.load("./data/trained_models/cifar_mixup_acc_90.36_ckpt.pth")["net"])

    model = VGG16()
    model.load_state_dict(torch.load("./data/trained_models/vgg_seed32_dropout.pkl"))

    model = model.cuda()
    model.eval()

    # file = 'adv_cifar'  # 不同的数据集
    # model_name = 'vgg'
    # dt = 'cifar10'

    # file = 'adv_cifar1'  # 不同的数据集
    # model_name = 'convcifar10'
    # dt = 'cifar10'

    # file = 'adv_svhn1'
    # model_name = 'alexnet'
    # dt = 'svhn'

    # file = 'adv_mnist1'
    # model_name = 'convmnist'
    # dt = 'mnist'

    file = 'adv_vgg'  # 不同的数据集
    model_name = 'vgg'
    dt = 'cifar10'

    # error_rate_list = [.01, .02, .04, .06, .08, .1]
    # error_rate_list = [0.01, .1, .2, .3, .4, .5, .6, .7, .8, .9]

    error_rate_list = [100, 200, 400, 600, 800, 1000]
    error_rate_list = [100, 1000, 2000]
    # dt = 'svhn'
    # model_name = 'alexnet'

    mode = 'FGSM'

    data_file = open('./{}/all_right_class_index.pkl'.format(file), 'rb')
    right_class = pickle.load(data_file)
    data_file.close()

    for error_rate in error_rate_list:
        # if error_rate == .0:
        #     adv_dt = torchvision.datasets.SVHN(root='./deephunter/models/data/',
        #                                         transform=transforms.ToTensor(), split='test')
        # else:
        # adv_dt, only_adv_dt, attack_list, attack_list_class = AA.PGD_test(model, device, dataset, right_class, eps=0.3, alpha=2 / 255, iters=40,
        #                       rate=error_rate)

        adv_dt, only_adv_dt, attack_list, attack_list_class = AA.FGSM_test(model, device, dataset, right_class, eps=0.5,
                              rate = error_rate)


            # for i in range(len(dataset)):
            #     if not (adv_dt.data[i]==dataset.data[i]).all():
            #         print("attack！！！！")


        root_path = "./{}/cluster_paths_{}/".format(file, error_rate)
        if os.path.exists(root_path) == False:
            os.makedirs(root_path)
        # np.save(root_path + '{}_{}_{}_{}.npy'.format(dt, model_name, mode, error_rate), adv_dt)
        pickle.dump(adv_dt, open(root_path + '{}_{}_{}_{}.pkl'.format(dt, model_name, mode, error_rate), 'wb'))
        print('save {}_{}_{}_{} at {} finished!'.format(dt, model_name, mode, error_rate, file))

        pickle.dump(only_adv_dt, open(root_path + '{}_only_{}_{}.pkl'.format(dt, mode, error_rate), 'wb'))
        pickle.dump(attack_list, open(root_path + '{}_attacklist_{}_{}.pkl'.format(dt, mode, error_rate), 'wb'))
        pickle.dump(attack_list_class, open(root_path + '{}_attackclass_{}_{}.pkl'.format(dt, mode, error_rate), 'wb'))
        print('save all finished!')
