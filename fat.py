from __future__ import print_function

import copy
import argparse
import torch
import torch.optim as optim
import torch.utils.data
from tensorboardX import SummaryWriter
from tqdm import tqdm
import warnings
from utils import eval_adv_test_whitebox, maxMarginLoss, \
    DatasetSplit, setup_seed, partition_data, average_weights, kl_adv, get_model, \
    save_checkpoint, maxMarginLoss_kl
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.train_loader = DataLoader(DatasetSplit(dataset, idxs),
                                       batch_size=self.args.local_bs, shuffle=True, num_workers=4)

    def pgd_train(self, model, criterion):
        model.train()
        if "cifar10" in args.dataset:
            weight_decay = 2e-3
        else:
            weight_decay = 2e-4
        optimizer = optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=weight_decay)
        criterion = maxMarginLoss(cls_num_list=args.cls_num_list, max_m=0.8, s=10, weight=None).cuda()
        criterion_kl = maxMarginLoss_kl(cls_num_list=args.cls_num_list, s=10, weight=None).cuda()
        for iter in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                optimizer.zero_grad()
                data, target = images.cuda(), labels.cuda()
                x_adv = kl_adv(model, data, cfgs, args.dataset, criterion_kl)
                model.train()
                output = model(x_adv)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        return model.state_dict()

def get_cls_num_list(traindata_cls_counts):
    cls_num_list = []
    num_class = 100 if args.dataset == "cifar100" else 10
    for key, val in traindata_cls_counts.items():
        temp = [0] * num_class  #
        for key_1, val_1 in val.items():
            temp[key_1] = val_1
        cls_num_list.append(temp)

    return cls_num_list


def train_local(local_model, server_model):
    # adversarial training
    criterion = maxMarginLoss(cls_num_list=[500] * 10, max_m=0.8, s=10, weight=None).cuda()
    w = local_model.pgd_train(server_model, criterion)

    return w


parser = argparse.ArgumentParser(description='Calibrated Federated Adversarial Training')
parser.add_argument('--epochs', type=int, default=76, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--beta', type=float, default=0.1,
                    help='dirichlet hyperparameter')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', default='ce',
                    help='directory of model for saving checkpoint')
parser.add_argument('--dataset', default='mnist', help='dataset')
parser.add_argument('--partition', default='dirichlet',
                    help='directory of model for saving checkpoint')
parser.add_argument('--model', default='cnn',
                    help='directory of model for saving checkpoint')
parser.add_argument('--comment', default='niid_beta_0.1',
                    help='directory of model for saving checkpoint')
parser.add_argument('--local_ep', type=int, default=1,
                    help="the number of local epochs: E")
parser.add_argument('--local_bs', type=int, default=128,
                    help="local batch size: B")
parser.add_argument('--num_users', type=int, default=5,
                    help="local batch size: B")

args = parser.parse_args()

sets = {"mnist": dict(step_size=0.01, epsilon=0.3),
        "fmnist": dict(step_size=0.01, epsilon=0.3),
        "cifar10": dict(step_size=2.0 / 255, epsilon=8.0 / 255),
        "svhn": dict(step_size=2.0 / 255, epsilon=8.0 / 255),
        "cifar100": dict(step_size=2.0 / 255, epsilon=8.0 / 255),
        }

cfgs = sets[args.dataset]

if __name__ == '__main__':

    setup_seed(2021)
    iid = "iid" if args.partition == "iid" else "niid"
    public = "{}/{}_{}_{}_{}".format(args.comment, args.model, args.dataset, iid, args.save)
    tf_writer = SummaryWriter(log_dir=public)
    train_dataset, test_dataset, user_groups, traindata_cls_counts, y_test = partition_data(
        args.num_users, args.dataset, args.partition, beta=args.beta)
    total = len(test_dataset)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256,
                                              shuffle=False, num_workers=4)
    # BUILD MODEL
    global_model = get_model(args)
    # global_model = CNNCifar().cuda() if args.model == "cnn" else ResNet18().cuda()
    description = "inference acc={:.4f}% loss={:.2f}, best_acc = {:.2f}%"

    global_model.train()
    cls_num_list = get_cls_num_list(traindata_cls_counts)
    print(cls_num_list)
    bst_cln_acc = -1
    bst_rob_acc = -1
    best_acc_ckpt = '{}/acc.pth'.format(public)
    best_asr_ckpt = '{}/asr.pth'.format(public)
    last_ckpt = '{}/last.pth'.format(public)
    # ===============================================
    for epoch in tqdm(range(args.epochs)):
        local_weights = []
        for idx in range(args.num_users):
            args.cls_num_list = cls_num_list[idx]

            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx])
            w = train_local(local_model, copy.deepcopy(global_model))
            local_weights.append(copy.deepcopy(w))

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)
        # evaluation on natural examples
        print('================================================================')
        natural_err_total, robust_err_total = eval_adv_test_whitebox(global_model, test_loader, cfgs, args.dataset)
        cln_acc, rob_acc = (total - natural_err_total) * 1.0 / total, (total - robust_err_total) * 1.0 / total
        print("Epoch:{},\t cln_acc:{}, \t rob_acc:{}".format(epoch, cln_acc, rob_acc))
        tf_writer.add_scalar('cln_acc', cln_acc, epoch)
        tf_writer.add_scalar('rob_acc', rob_acc, epoch)
        save_checkpoint({
            'state_dict': global_model.state_dict(),
            'epoch': epoch,
        }, cln_acc > bst_cln_acc, best_acc_ckpt)

        save_checkpoint({
            'state_dict': global_model.state_dict(),
            'epoch': epoch,
        }, rob_acc > bst_rob_acc, best_asr_ckpt)
        bst_cln_acc = max(bst_cln_acc, cln_acc)
        bst_rob_acc = max(bst_rob_acc, rob_acc)
        save_checkpoint({
            'state_dict': global_model.state_dict(),
            'epoch': epoch,
        }, 1 > 0, last_ckpt)
        print('================================================================')