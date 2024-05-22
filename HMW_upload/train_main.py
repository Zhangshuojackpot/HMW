from __future__ import print_function
import sys
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import random
import os
import argparse
import numpy as np
from PreResNet_cifar_real import *
import dataloader_cifar as dataloader
from Contrastive_loss import *
from RANDWSTLAN_TOP_INRM_MIX_BCER import NaturalDistanceWeighting, SCELoss
import itertools



## Arguments to pass 
parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode', default='sym')
parser.add_argument('--num_epochs', default=250, type=int)
parser.add_argument('--tau', default=5, type=float, help='filtering coefficient')
parser.add_argument('--r', default=0.8, type=float, help='noise ratio')
parser.add_argument('--d_u', default=0.7, type=float)
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=30, type=float, help='weight for unsupervised loss')
parser.add_argument('--lambda_c', default=0.025, type=float, help='weight for contrastive loss')
parser.add_argument('--metric', type=str, default='JSD', help='Comparison Metric')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--resume', default=False, type=bool, help='Resume from the checkpoint')
parser.add_argument('--num_class', default=100, type=int)
parser.add_argument('--data_path', default='./data/cifar100', type=str, help='path to dataset: ./data/cifar10, ./data/cifar100')
parser.add_argument('--dataset', default='cifar100', type=str, help='cifar10, cifar100')
parser.add_argument('--method', default='hmw+cce', type=str, help='hmw+cce, hmw+sce, hmw+uc, cce, sce, uc')
parser.add_argument('--alpha_nmw', default=100., type=float, help='weight of AUM')
parser.add_argument('--beta_nmw', default=100., type=float, help='weight of TKUM')
parser.add_argument('--top_rate_nmw', default=0.01, type=float, help='the rate of selected top logits')
parser.add_argument('--milestones', default='125/200', type=str, help='epoch to change the learning rate')

parser.add_argument('--if_aum', default=1, type=float, help='if using AUM')
parser.add_argument('--if_anneal', default=1, type=float, help='if using anneal strategy')
parser.add_argument('--if_spherical', default=1, type=float, help='if changing to hyperspherical space')


parser.add_argument('--check_loc', default='./checkpoint_cifar10_cce_final_ndw_sym_real/', type=str, help='path of saving results')

args = parser.parse_args()


print('Training ' + args.method)

## GPU Setup
torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
## Download the Datasets
if args.dataset == 'cifar10':
    torchvision.datasets.CIFAR10(args.data_path, train=True, download=True)
    torchvision.datasets.CIFAR10(args.data_path, train=False, download=True)
else:
    torchvision.datasets.CIFAR100(args.data_path, train=True, download=True)
    torchvision.datasets.CIFAR100(args.data_path, train=False, download=True)

## Checkpoint Location
folder = args.dataset + '_' + args.noise_mode + '_' + args.method
if not os.path.exists(args.check_loc):
    os.mkdir(args.check_loc)
model_save_loc_f = args.check_loc + str(args.r)
if not os.path.exists(model_save_loc_f):
    os.mkdir(model_save_loc_f)
model_save_loc = model_save_loc_f + '/' + folder
if not os.path.exists(model_save_loc):
    os.mkdir(model_save_loc)

## Log files
stats_log = open(model_save_loc + '/%s_%.1f_%s' % (args.dataset, args.r, args.noise_mode) + '_stats.txt', 'a')
test_log = open(model_save_loc + '/%s_%.1f_%s' % (args.dataset, args.r, args.noise_mode) + '_acc.txt', 'a')
test_loss_log = open(model_save_loc + '/test_loss.txt', 'a')
train_acc = open(model_save_loc + '/train_acc.txt', 'a')
train_loss = open(model_save_loc + '/train_loss.txt', 'a')


# SSL-Training
def train(epoch, net, weighting, net2, weighting2, optimizer, labeled_trainloader, unlabeled_trainloader):
    net2.eval()  # Freeze one network and train the other
    net.train()

    if weighting2 != None:
        weighting2.eval()
    if weighting != None:
        weighting.train()

    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1

    ## Loss statistics
    loss_x = 0
    loss_u = 0
    loss_scl = 0
    loss_ucl = 0

    for batch_idx, (inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x_int, w_x, l_idxs) in enumerate(
            labeled_trainloader):
        try:
            inputs_u, inputs_u2, inputs_u3, inputs_u4, ul_idxs = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2, inputs_u3, inputs_u4, ul_idxs = next(unlabeled_train_iter)

        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x_int.view(-1, 1), 1)
        w_x = w_x.view(-1, 1).type(torch.FloatTensor)

        inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), inputs_x3.cuda(), inputs_x4.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2, inputs_u3, inputs_u4 = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda(), inputs_u4.cuda()
        labels_x_int = labels_x_int.cuda()

        with torch.no_grad():
            # Label co-guessing of unlabeled samples
            _, or_feats_u11, outputs_u11 = net(inputs_u)
            if weighting != None:
                outputs_u11 = weighting(or_feats_u11)


            _, or_feats_u12, outputs_u12 = net(inputs_u2)
            if weighting != None:
                outputs_u12 = weighting(or_feats_u12)


            _, or_feats_u21, outputs_u21 = net2(inputs_u)
            if weighting != None:
                outputs_u21 = weighting2(or_feats_u21)


            _, or_feats_u22, outputs_u22 = net2(inputs_u2)
            if weighting != None:
                outputs_u22 = weighting2(or_feats_u22)


            ## Pseudo-label
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21,
                                                                                                        dim=1)
                  + torch.softmax(outputs_u22, dim=1)) / 4

            ptu = pu ** (1 / args.T)  ## Temparature Sharpening

            targets_u = ptu / ptu.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

            ## Label refinement
            _, or_feats_x, outputs_x = net(inputs_x)
            if weighting != None:
                outputs_x = weighting(or_feats_x)

            _, or_feats_x2, outputs_x2 = net(inputs_x2)
            if weighting != None:
                outputs_x2 = weighting(or_feats_x2)

            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2

            px = w_x * labels_x + (1 - w_x) * px
            ptx = px ** (1 / args.T)  ## Temparature sharpening

            targets_x = ptx / ptx.sum(dim=1, keepdim=True)
            targets_x = targets_x.detach()

        ## Unsupervised Contrastive Loss
        f1, _f, _ = net(inputs_u3)
        f2, _f, _ = net(inputs_u4)
        f1 = F.normalize(f1, dim=1)
        f2 = F.normalize(f2, dim=1)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        loss_simCLR = contrastive_criterion(features)

        # MixMatch
        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1 - l)

        all_inputs = torch.cat([inputs_x3, inputs_x4, inputs_u3, inputs_u4], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)
        all_labels = torch.cat([torch.argmax(targets_x, dim=-1), torch.argmax(targets_x, dim=-1), torch.argmax(targets_u, dim=-1), torch.argmax(targets_u, dim=-1)], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        original_labels_a, original_labels_b = all_labels, all_labels[idx]

        ## Mixup
        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        _, or_feats, logits = net(mixed_input)
        weights = None
        if weighting != None:
            logits, weights = weighting(or_feats, labels=mixed_target, ep=epoch, mixup=True, labels_a=original_labels_a, labels_b=original_labels_b, mix_rate=l)

        logits_x = logits[:batch_size * 2]
        mixed_target_x = mixed_target[:batch_size * 2]

        logits_u = logits[batch_size * 2:]
        mixed_target_u = mixed_target[batch_size * 2:]


        ## Combined Loss
        Lx, Lu, lamb = criterion(logits_x, mixed_target_x, logits_u, mixed_target_u,
                                 epoch + batch_idx / num_iter, warm_up, weights)

        ## Regularization
        prior = torch.ones(args.num_class) / args.num_class
        prior = prior.cuda()

        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))

        ## Total Loss
        loss = Lx + lamb * Lu + args.lambda_c * loss_simCLR + penalty

        ## Accumulate Loss
        loss_x += Lx.item()
        loss_u += Lu.item()
        # loss_ucl += loss_simCLR.item()

        # Compute gradient and Do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    sys.stdout.write('\r')
    sys.stdout.write(
        '%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f Contrastive Loss:%.4f lr:%.8f'
        % (args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx + 1, num_iter,
           loss_x / (batch_idx + 1), loss_u / (batch_idx + 1),
           loss_ucl / (batch_idx + 1), optimizer.param_groups[0]['lr']))
    sys.stdout.flush()


## For Standard Training
def standard(epoch, net, optimizer, dataloader, weighting=None):
    net.train()
    if weighting != None:
        weighting.train()
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    # batch_idx = 0

    total = 0
    correct = 0


    for batch_idx, (inputs, labels, path) in enumerate(dataloader):
        # for inputs, labels, path in dataloader:
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        _, or_feats, outputs = net(inputs)
        if weighting != None:
            outputs, w_ = weighting(or_feats, labels, idx=path, ep=epoch)
            if epoch < warm_up:
                loss = USEloss(outputs, labels.long()).squeeze().mean()
            else:
                loss = (USEloss(outputs, labels.long()).squeeze() * w_.squeeze()).mean()

        pre_out = torch.softmax(outputs, dim=1)
        __, predicted = torch.max(pre_out, 1)


        if args.noise_mode == 'asym':  # Penalize confident prediction for asymmetric noise
            penalty = conf_penalty(outputs)
            L = loss + penalty
        else:
            L = loss

        total += labels.size(0)
        correct += predicted.eq(labels).cpu().sum().item()

        acc = 100. * correct / total

        L.backward()
        optimizer.step()

    sys.stdout.write('\r')
    sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f acc: %.4f lr: %.8f'
                     % (args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx + 1, num_iter,
                        loss.item(), acc, optimizer.param_groups[0]['lr']))
    sys.stdout.flush()
    train_acc.write('ep' + str(epoch) + ':' + str(acc) + '\n')
    train_acc.flush()



## For Training Accuracy
def warmup_val(epoch, net, optimizer, dataloader):
    net.train()
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    total = 0
    correct = 0
    loss_x = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels, path) in enumerate(dataloader):
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            _, outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            loss = USEloss(outputs, labels.long())
            loss_x += loss.item()

            total += labels.size(0)
            correct += predicted.eq(labels).cpu().sum().item()

    acc = 100. * correct / total
    print("\n| Train Epoch #%d\t Accuracy: %.2f%%\n" % (epoch, acc))

    train_loss.write(str(loss_x / (batch_idx + 1)))
    train_acc.write(str(acc))
    train_acc.flush()
    train_loss.flush()

    return acc


## Test Accuracy
def test(epoch, net1, net2, weighting1, weighting2):
    net1.eval()
    if weighting1 != None:
        weighting1.eval()
    net2.eval()
    if weighting2 != None:
        weighting2.eval()

    num_samples = 1000
    correct = 0
    total = 0
    loss_x = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            _, or_feats1, outputs1 = net1(inputs)
            if weighting1 != None:
                outputs1 = weighting1(or_feats1)
            _, or_feats2, outputs2 = net2(inputs)
            if weighting2 != None:
                outputs2 = weighting2(or_feats2)
            outputs1 = torch.softmax(outputs1, dim=1)
            outputs2 = torch.softmax(outputs2, dim=1)
            outputs = (outputs1 + outputs2) / 2
            _, predicted = torch.max(outputs, 1)
            loss = USEloss(outputs, targets.long()).mean()
            loss_x += loss.item()

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()

    acc = 100. * correct / total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" % (epoch, acc))
    test_log.write('ep' + str(epoch) + ':' + str(acc) + '\n')
    test_log.flush()
    test_loss_log.write(str(loss_x / (batch_idx + 1)) + '\n')
    test_loss_log.flush()
    return acc


## Test Accuracy
def test_onenet(epoch, net1, weighting=None):
    net1.eval()
    if weighting != None:
        weighting.eval()

    num_samples = 1000
    correct = 0
    total = 0
    loss_x = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            _, or_feats1, outputs1 = net1(inputs)
            if weighting != None:
                outputs1 = weighting(or_feats1)
            outputs = outputs1
            pre_out = torch.softmax(outputs, dim=1)
            __, predicted = torch.max(pre_out, 1)
            loss = USEloss(outputs, targets.long()).mean()
            loss_x += loss.item()

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()

    acc = 100. * correct / total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" % (epoch, acc))
    test_log.write('ep' + str(epoch) + ':' + str(acc) + '\n')
    test_log.flush()
    test_loss_log.write(str(loss_x / (batch_idx + 1)) + '\n')
    test_loss_log.flush()
    return acc


# KL divergence
def kl_divergence(p, q):
    return (p * ((p + 1e-10) / (q + 1e-10)).log()).sum(dim=1)


## Jensen-Shannon Divergence
class Jensen_Shannon(nn.Module):
    def __init__(self):
        super(Jensen_Shannon, self).__init__()
        pass

    def forward(self, p, q):
        m = (p + q) / 2
        return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


## Calculate JSD
def Calculate_JSD(model1, weighting1, model2, weighting2, num_samples):
    JS_dist = Jensen_Shannon()
    JSD = torch.zeros(num_samples)

    for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        batch_size = inputs.size()[0]

        ## Get outputs of both network
        with torch.no_grad():
            if weighting1 == None:
                logit1 = model1(inputs)[2]
            else:
                logit1 = weighting1(model1(inputs)[1])
            out1 = torch.nn.Softmax(dim=1).cuda()(logit1)

            if weighting2 == None:
                logit2 = model2(inputs)[2]
            else:
                logit2 = weighting2(model2(inputs)[1])
            out2 = torch.nn.Softmax(dim=1).cuda()(logit2)

        ## Get the Prediction
        out = (out1 + out2) / 2

        ## Divergence clculator to record the diff. between ground truth and output prob. dist.
        dist = JS_dist(out, F.one_hot(targets, num_classes=args.num_class))
        JSD[int(batch_idx * batch_size):int((batch_idx + 1) * batch_size)] = dist

    return JSD


## Unsupervised Loss coefficient adjustment
def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u * float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up, weights):

        if weights != None:
            w_x = weights[:outputs_x.shape[0]]
            w_u = weights[outputs_x.shape[0]:]

            probs_u = torch.softmax(outputs_u, dim=1)
            Lx = -torch.mean(w_x.squeeze() * torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1).squeeze())
            Lu = torch.mean(w_u * ((probs_u - targets_u) ** 2))
        else:
            probs_u = torch.softmax(outputs_u, dim=1)
            Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
            Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, linear_rampup(epoch, warm_up)


class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


## Choose Warmup period based on Dataset
num_samples = 50000
if args.dataset == 'cifar10':
    warm_up = 10
    #warm_up = 10
elif args.dataset == 'cifar100':
    warm_up = 30
    #warm_up = 10


def create_model():
    model = ResNet18(num_classes=args.num_class)
    model = model.cuda()
    if '+' in args.method:
        if 'hmw' in args.method:
            print('training hmw!')
            weighting = NaturalDistanceWeighting(num_classes=args.num_class, feat_dim=512,
                                                 train_size=num_samples, train_epoch=args.num_epochs,
                                                 warmup_epoch=warm_up, alpha=args.alpha_nmw,
                                                 beta=args.beta_nmw, top_rate=args.top_rate_nmw,
                                                 if_aum=args.if_aum, if_anneal=args.if_anneal, if_spherical = args.if_spherical)
        else:
            assert 1 == 0
        weighting = weighting.cuda()
    else:
        weighting = None

    return model, weighting


## Call the dataloader
loader = dataloader.cifar_dataloader(args.dataset, r=args.r, noise_mode=args.noise_mode, batch_size=args.batch_size,
                                     num_workers=0, \
                                     root_dir=model_save_loc, data_dir=args.data_path, log=stats_log,
                                     noise_file='%s/clean_%.4f_%s.npz' % (args.data_path, args.r, args.noise_mode))

print('| Building net')
net1, weighting1 = create_model()
net2, weighting2 = create_model()
cudnn.benchmark = True

## Semi-Supervised Loss
criterion = SemiLoss()

## Optimizer and Scheduler

if weighting1 == None and weighting2 == None:
    params1 = net1.parameters()
    params2 = net2.parameters()
else:
    params1 = itertools.chain(net1.parameters(), weighting1.parameters())
    params2 = itertools.chain(net2.parameters(), weighting2.parameters())

optimizer1 = optim.SGD(params1, lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(params2, lr=args.lr, momentum=0.9, weight_decay=5e-4)
#optimizer1 = optim.Adam(params1, weight_decay=5e-4)
#optimizer2 = optim.Adam(params2, weight_decay=5e-4)

#scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, 280, 2e-4)
#scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, 280, 2e-4)

scheduler1 = optim.lr_scheduler.MultiStepLR(optimizer1, milestones=[int(args.milestones.split('/')[0]), int(args.milestones.split('/')[1])], gamma=0.1)
scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer2, milestones=[int(args.milestones.split('/')[0]), int(args.milestones.split('/')[1])], gamma=0.1)

## Loss Functions
CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss(reduction='none')
MSE_loss = nn.MSELoss(reduction='none')
contrastive_criterion = SupConLoss()

if 'cce' in args.method or 'uc' in args.method:
    USEloss = CEloss
    print('using cce\n')
elif 'sce' in args.method:
    print('using sce\n')
    if args.dataset == 'cifar10':
        #USEloss = NCEandRCE(alpha=1, beta=1, num_classes=10)
        USEloss = SCELoss(a=0.1, b=1, num_classes=10)
    elif args.dataset == 'cifar100':
        #USEloss = NCEandRCE(alpha=10, beta=1, num_classes=100)
        USEloss = SCELoss(a=6, b=0.1, num_classes=100)
    else:
        assert 1 == 0


if args.noise_mode == 'asym':
    conf_penalty = NegEntropy()

## Resume from the warmup checkpoint
model_name_1 = 'Net1.pth'
model_name_2 = 'Net2.pth'

if args.resume:
    if 'uc' in args.method:
        check1 = torch.load(os.path.join(model_save_loc, model_name_1))
        check2 = torch.load(os.path.join(model_save_loc, model_name_2))

        net1.load_state_dict(check1['net'])
        weighting1.load_state_dict(check1['weighting'])

        net2.load_state_dict(check2['net'])
        weighting2.load_state_dict(check2['weighting'])

        start_epoch = np.minimum(check1['epoch'], check2['epoch']) + 1

    else:
        check1 = torch.load(os.path.join(model_save_loc, model_name_1))

        net1.load_state_dict(check1['net'])
        weighting1.load_state_dict(check1['weighting'])

        start_epoch = check1['epoch'] + 1

    scheduler1.step(start_epoch)
    scheduler2.step(start_epoch)

else:
    start_epoch = 0

# best_acc = 0

## Warmup and SSL-Training
for epoch in range(start_epoch, args.num_epochs + 1):
    test_loader = loader.run(0, 'test')
    eval_loader = loader.run(0, 'eval_train')
    # trainloader = loader.run(0,'warmup')

    if 'uc' in args.method:

        ## Warmup Stage
        if epoch < warm_up:
            trainloader = loader.run(0, 'warmup')

            print('Warmup Model1')
            standard(epoch, net1, optimizer1, trainloader, weighting=weighting1)

            print('\nWarmup Model2')
            standard(epoch, net2, optimizer2, trainloader, weighting=weighting2)

        else:
            ## Calculate JSD values and Filter Rate
            prob = Calculate_JSD(net2, weighting2, net1, weighting1, num_samples)
            threshold = torch.mean(prob)
            if threshold.item() > args.d_u:
                threshold = threshold - (threshold - torch.min(prob)) / args.tau
            SR = torch.sum(prob < threshold).item() / num_samples

            print('Train Net1\n')
            labeled_trainloader, unlabeled_trainloader = loader.run(SR, 'train', prob=prob)  # Uniform Selection
            train(epoch, net1, weighting1, net2, weighting2, optimizer1, labeled_trainloader,
                  unlabeled_trainloader)  # train net1

            ## Calculate JSD values and Filter Rate
            prob = Calculate_JSD(net2, weighting2, net1, weighting1, num_samples)
            threshold = torch.mean(prob)
            if threshold.item() > args.d_u:
                threshold = threshold - (threshold - torch.min(prob)) / args.tau
            SR = torch.sum(prob < threshold).item() / num_samples

            print('\nTrain Net2')
            labeled_trainloader, unlabeled_trainloader = loader.run(SR, 'train', prob=prob)  # Uniform Selection
            train(epoch, net2, weighting2, net1, weighting1, optimizer2, labeled_trainloader,
                  unlabeled_trainloader)  # train net1

        acc = test(epoch, net1, net2, weighting1, weighting2)
        scheduler1.step()
        scheduler2.step()

        model_name_1 = 'Net1.pth'
        model_name_2 = 'Net2.pth'
        '''
        print("Save the Model-----")
        if weighting1 == None:
            checkpoint1 = {
                'net': net1.state_dict(),
                'Model_number': 1,
                'Noise_Ratio': args.r,
                'Loss Function': 'CrossEntropyLoss',
                'Optimizer': 'SGD',
                'Noise_mode': args.noise_mode,
                'Accuracy': acc,
                'Dataset': args.dataset,
                'Batch Size': args.batch_size,
                'epoch': epoch,
            }
        else:
            checkpoint1 = {
                'net': net1.state_dict(),
                'weighting': weighting1.state_dict(),
                'Model_number': 1,
                'Noise_Ratio': args.r,
                'Loss Function': 'CrossEntropyLoss',
                'Optimizer': 'SGD',
                'Noise_mode': args.noise_mode,
                'Accuracy': acc,
                'Dataset': args.dataset,
                'Batch Size': args.batch_size,
                'epoch': epoch,
            }

        if weighting2 == None:
            checkpoint2 = {
                'net': net2.state_dict(),
                'Model_number': 2,
                'Noise_Ratio': args.r,
                'Loss Function': 'CrossEntropyLoss',
                'Optimizer': 'SGD',
                'Noise_mode': args.noise_mode,
                'Accuracy': acc,
                'Dataset': args.dataset,
                'Batch Size': args.batch_size,
                'epoch': epoch,
            }
        else:
            checkpoint2 = {
                'net': net2.state_dict(),
                'weighting': weighting2.state_dict(),
                'Model_number': 2,
                'Noise_Ratio': args.r,
                'Loss Function': 'CrossEntropyLoss',
                'Optimizer': 'SGD',
                'Noise_mode': args.noise_mode,
                'Accuracy': acc,
                'Dataset': args.dataset,
                'Batch Size': args.batch_size,
                'epoch': epoch,
            }

        torch.save(checkpoint1, os.path.join(model_save_loc, model_name_1))
        torch.save(checkpoint2, os.path.join(model_save_loc, model_name_2))
        '''
        # best_acc = acc
    else:
        trainloader = loader.run(0, 'warmup')


        print('Train Model1')
        standard(epoch, net1, optimizer1, trainloader, weighting=weighting1)
        acc = test_onenet(epoch, net1, weighting=weighting1)
        scheduler1.step()

        # if acc > best_acc:

        model_name_1 = 'Net1.pth'
        # model_name_2 = 'Net2.pth'
        '''
        print("Save the Model-----")
        if weighting1 == None:
            checkpoint1 = {
                'net': net1.state_dict(),
                'Model_number': 1,
                'Noise_Ratio': args.r,
                'Loss Function': 'CrossEntropyLoss',
                'Optimizer': 'SGD',
                'Noise_mode': args.noise_mode,
                'Accuracy': acc,
                'Dataset': args.dataset,
                'Batch Size': args.batch_size,
                'epoch': epoch,
            }
        else:
            checkpoint1 = {
                'net': net1.state_dict(),
                'weighting': weighting1.state_dict(),
                'Model_number': 1,
                'Noise_Ratio': args.r,
                'Loss Function': 'CrossEntropyLoss',
                'Optimizer': 'SGD',
                'Noise_mode': args.noise_mode,
                'Accuracy': acc,
                'Dataset': args.dataset,
                'Batch Size': args.batch_size,
                'epoch': epoch,
            }

        torch.save(checkpoint1, os.path.join(model_save_loc, model_name_1))
        # torch.save(checkpoint2, os.path.join(model_save_loc, model_name_2))
        # best_acc = acc
        '''


