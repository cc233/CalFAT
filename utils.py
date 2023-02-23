import copy
import numpy as np
import numpy.random
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, TensorDataset
from torchvision import datasets, transforms
from nets import CNNCifar, CNNCifar100, CNNMnist

data_dir = './data'

def load_data(dataset):
    if dataset == "mnist":
        train_dataset = datasets.MNIST(data_dir, train=True,
                                       transform=transforms.Compose(
                                           [transforms.ToTensor()]))
        test_dataset = datasets.MNIST(data_dir, train=False,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                      ]))
    elif dataset == "fmnist":
        train_dataset = datasets.FashionMNIST(data_dir, train=True,
                                              transform=transforms.Compose(
                                                  [transforms.ToTensor()]))
        test_dataset = datasets.FashionMNIST(data_dir, train=False,
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                             ]))
    elif dataset == "svhn":
        train_dataset = datasets.SVHN(data_dir, split="train",
                                      transform=transforms.Compose(
                                          [transforms.ToTensor()]),
                                      download=True)
        test_dataset = datasets.SVHN(data_dir, split="test",
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                     ]),
                                      download=True)
    elif dataset == "cifar10":
        train_dataset = datasets.CIFAR10(data_dir, train=True,
                                         transform=transforms.Compose(
                                             [
                                                 transforms.RandomCrop(32, padding=4),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                             ]))
        test_dataset = datasets.CIFAR10(data_dir, train=False,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                        ]))
    elif dataset == "cifar100":
        train_dataset = datasets.CIFAR100(data_dir, train=True,
                                          transform=transforms.Compose(
                                              [
                                                  transforms.RandomCrop(32, padding=4),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                              ]))
        test_dataset = datasets.CIFAR100(data_dir, train=False,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                         ]))
    
        return X_train, y_train, X_test, y_test, train_dataset, test_dataset
    else:
        raise NotImplementedError

    if dataset == "svhn":
        X_train, y_train = train_dataset.data, train_dataset.labels
        X_test, y_test = test_dataset.data, test_dataset.labels
    else:
        X_train, y_train = train_dataset.data, train_dataset.targets
        X_test, y_test = test_dataset.data, test_dataset.targets
    if "cifar10" in dataset or dataset == "svhn":
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
    else:
        X_train = X_train.data.numpy()
        y_train = y_train.data.numpy()
        X_test = X_test.data.numpy()
        y_test = y_test.data.numpy()

    return X_train, y_train, X_test, y_test, train_dataset, test_dataset

def record_net_data_stats(y_train, net_dataidx_map, dataset):
    net_cls_counts = {}
    num_class = np.unique(y_train).shape[0]
    for net_i, dataidx in net_dataidx_map.items():  # label:sets
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        for i in range(num_class):
            if i in tmp.keys():
                continue
            else:
                tmp[i] = 1  # 5

        net_cls_counts[net_i] = tmp

    print('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts

def partition_data(n_user, dataset, partition, beta=0.4):
    n_parties = n_user
    X_train, y_train, X_test, y_test, train_dataset, test_dataset = load_data(dataset)
    data_size = y_train.shape[0]

    if partition == "iid":
        idxs = numpy.random.permutation(data_size)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    elif partition == "dirichlet":
        min_size = 0
        min_require_size = 10
        label = np.unique(y_test).shape[0]
        net_dataidx_map = {}

        idx_batch = [[] for _ in range(n_parties)]
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(label):
                idx_k = np.where(y_train == k)[0]
                numpy.random.shuffle(idx_k)  # shuffle the label
                proportions = numpy.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array(  # 0 or x
                    [p * (len(idx_j) < data_size / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_parties):
            numpy.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition > "label0" and partition <= "label9":
        num = eval(partition[5:])
        num_shards, num_imgs = n_parties * num, int(data_size / (n_parties * num))
        idx_shard = [i for i in range(num_shards)]
        net_dataidx_map = {i: np.ndarray([0], dtype=np.int64) for i in range(n_parties)}
        idxs = np.arange(num_shards * num_imgs)
        labels = y_train[:idxs.shape[0]]

        # sort labels
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]

        for i in range(n_parties):
            rand_set = set(numpy.random.choice(idx_shard, num, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                net_dataidx_map[i] = np.concatenate(
                    (net_dataidx_map[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    else:
        raise Exception('invalid partition')

    train_data_cls_counts = record_net_data_stats(y_train, net_dataidx_map, dataset)

    return train_dataset, test_dataset, net_dataidx_map, train_data_cls_counts, y_test

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        if 'num_batches_tracked' in key:
            w_avg[key] = w_avg[key].true_divide(len(w))
        else:
            w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True

class maxMarginLoss_kl(nn.Module):
    def __init__(self, cls_num_list, weight=None, s=30):
        super(maxMarginLoss_kl, self).__init__()
        m_list = torch.FloatTensor(cls_num_list).cuda()
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x):
        output = x + 0.1 * torch.log(self.m_list + 1e-7)
        return output

class maxMarginLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(maxMarginLoss, self).__init__()
        m_list = torch.FloatTensor(cls_num_list).cuda()
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        output = x + 0.1 * torch.log(self.m_list + 1e-7)
        return F.cross_entropy(self.s * output, target, weight=self.weight)

def kl_adv(model, x_natural, cfgs, dataset, marloss):
    epsilon, step_size = cfgs["epsilon"], cfgs["step_size"]
    num_steps = 40 if dataset == "mnist" else 10
    criterion_kl = nn.KLDivLoss(size_average=False)
    # marloss = maxMarginLoss_kl()
    model.eval()
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    for _ in range(num_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(marloss(model(x_adv)), dim=1),
                                   F.softmax(marloss(model(x_natural)), dim=1))
            # loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
            #                        F.softmax(model(x_natural), dim=1))
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    return x_adv

def _pgd_whitebox(model, X, y, cfgs, dataset):
    epsilon, step_size = cfgs["epsilon"], cfgs["step_size"]
    num_steps = 40 if dataset == "mnist" else 20
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)

    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda()
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    return err, err_pgd

def eval_adv_test_whitebox(model, test_loader, cfgs, dataset):
    model.eval()
    robust_err_total = 0
    natural_err_total = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_whitebox(model, X, y, cfgs, dataset)
        robust_err_total += err_robust
        natural_err_total += err_natural
    return natural_err_total, robust_err_total

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    if is_best:
        torch.save(state, filename)

def get_model(args):
    if args.dataset == "mnist":
        global_model = CNNMnist(n_classes=10).cuda()
    elif args.dataset == "fmnist":
        global_model = CNNMnist(n_classes=10).cuda()
    elif args.dataset == "cifar10":
        global_model = CNNCifar().cuda()
    elif args.dataset == "svhn":
        global_model = CNNCifar().cuda()
    elif args.dataset == "cifar100":
        global_model = CNNCifar100().cuda()
    else:
        raise Exception('invalid model')

    return global_model
