
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import sys
import math

eps = 1e-7

class SCELoss(nn.Module):
    def __init__(self, num_classes=10, a=1, b=1):
        super(SCELoss, self).__init__()
        self.num_classes = num_classes
        self.a = a
        self.b = b
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, labels):
        ce = self.cross_entropy(pred, labels)
        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=eps, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))

        loss = self.a * ce + self.b * rce
        return loss


class NaturalDistanceWeighting(nn.Module):

    def __init__(self, num_classes, feat_dim, train_size, train_epoch, warmup_epoch, alpha=10., beta=2., top_rate=0.1, bias=False,
                 if_aum=0, if_anneal=0, if_spherical=0) -> None:
        super(NaturalDistanceWeighting, self).__init__()
        self.feat_dim = feat_dim
        self.top_rate = top_rate
        self.num_classes = num_classes
        self.train_epoch = train_epoch
        self.warmup_epoch = warmup_epoch
        self.add_weights = torch.zeros(train_epoch, train_size).cuda()
        #self.alpha = nn.Parameter(torch.randn(1, 1))
        #self.beta = nn.Parameter(torch.randn(1, 1))
        #self.a = nn.Parameter(torch.randn(1, 1))
        self.alpha = alpha
        self.beta = beta
        self.if_aum = if_aum
        self.if_anneal = if_anneal
        self.if_spherical = if_spherical

        #print('topk=', np.maximum(int((self.num_classes - 2) * self.top_rate), 1))
        self.weight = nn.Parameter(torch.empty((num_classes, feat_dim)))
        if bias:
            self.bias = nn.Parameter(torch.empty(num_classes))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        print('alpha=', self.alpha)
        print('beta=', self.beta)
        #print('s=', self.s)
        print('top_k=',np.maximum(int((self.num_classes - 2) * self.top_rate), 1))
        print('if_aum={}, if_anneal={}, if_spherical={}'.format(self.if_aum, self.if_anneal, self.if_spherical))

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)


    def forward(self, feat, labels=None, idx=None, ep=None, mixup=False, labels_a=None, labels_b=None, mix_rate=None):

        logits = F.linear(feat, self.weight, self.bias)
        s_logits = torch.softmax(logits, dim=-1)

        nfeat = F.normalize(feat, dim=1)
        nmeans = F.normalize(self.weight, dim=1)

        feat_dis = nfeat.unsqueeze(dim=1)
        means_dis = nmeans.unsqueeze(dim=0)

        if self.if_spherical == 1:
            angle_distances = (- torch.sum(feat_dis * means_dis, dim=-1) + 1.) / 2
        else:
            angle_distances = - F.softmax(logits, dim=-1)

        if labels is None:
            return logits


        else:
            if mixup == False:
                labeled_angle_distance = torch.gather(angle_distances, dim=-1, index=labels.unsqueeze(dim=-1))

                # min_angle_distance = torch.min(angle_distances, dim=-1, keepdim=True).values

                arrange_idxs = torch.arange(0, self.num_classes).unsqueeze(dim=0).tile((len(labels), 1)).cuda()

                other_idxs = torch.empty(len(labels), self.num_classes - 1, dtype=torch.int64).cuda()

                for i in range(len(labels)):
                    other_idxs[i] = arrange_idxs[i][arrange_idxs[i] != labels[i]]

                other_angle_distances = torch.gather(angle_distances, dim=-1, index=other_idxs)

                min_other_angle = torch.min(other_angle_distances, dim=-1, keepdim=True)

                min_other_angle_distance = min_other_angle.values
                min_other_angle_idxs = min_other_angle.indices

                other_other_idxs = torch.empty(len(labels), self.num_classes - 2, dtype=torch.int64).cuda()

                for i in range(len(labels)):
                    other_other_idxs[i] = torch.cat(
                        (other_idxs[i][0: min_other_angle_idxs[i]], other_idxs[i][min_other_angle_idxs[i] + 1:]), dim=0)

                other_other_angle_distances = torch.gather(angle_distances, dim=-1, index=other_other_idxs)

                top_k = np.maximum(int((self.num_classes - 2) * self.top_rate), 1)

                other_other_topk = torch.sort(other_other_angle_distances, dim=-1).values[:, :top_k]

                other_other_average_distances = other_other_topk.mean(dim=-1)

                assert torch.min(other_other_average_distances.squeeze() - min_other_angle_distance.squeeze()) >= 0

                if self.if_aum == 1:
                    weights = torch.sigmoid(self.alpha * (min_other_angle_distance.squeeze() - labeled_angle_distance.squeeze())) \
                              + torch.exp(self.beta * (min_other_angle_distance.squeeze() - other_other_average_distances.squeeze()))

                    weights = weights / 2.

                else:
                    weights = torch.exp(self.beta * (min_other_angle_distance.squeeze() - other_other_average_distances.squeeze()))


                self.add_weights[ep, idx] = weights.squeeze().detach()

                descends = torch.softmax(
                    torch.tensor([1 + np.cos((i / self.train_epoch) * np.pi) for i in range(ep + 1)]), dim=0).cuda().unsqueeze(dim=-1)

                logits = F.normalize(logits, dim=-1)

                #final_weights = (self.add_weights[:, idx] / (ep + 1)).squeeze().unsqueeze(dim=-1)
                if self.if_anneal == 1:
                    final_weights = (self.add_weights[:ep + 1, idx] * descends).sum(dim=0).squeeze().unsqueeze(dim=-1)
                else:
                    final_weights = weights.squeeze().unsqueeze(dim=-1)

            else:
                labeled_angle_distance_a = torch.gather(angle_distances, dim=-1, index=labels_a.unsqueeze(dim=-1))
                labeled_angle_distance_b = torch.gather(angle_distances, dim=-1, index=labels_b.unsqueeze(dim=-1))
                # labeled_angle_distance = mix_rate * labeled_angle_distance_a + (1 -mix_rate) * labeled_angle_distance_b
                labeled_angle_distance = torch.minimum(labeled_angle_distance_a, labeled_angle_distance_b)

                # min_angle_distance = torch.min(angle_distances, dim=-1, keepdim=True).values

                arrange_idxs = torch.arange(0, self.num_classes).unsqueeze(dim=0).tile((len(labels), 1)).cuda()

                other_idxs = torch.empty(len(labels), self.num_classes - 2, dtype=torch.int64).cuda()

                for i in range(len(labels)):
                    other_idx = arrange_idxs[i][arrange_idxs[i] != labels_a[i]]
                    other_idx = other_idx[other_idx != labels_b[i]]
                    if len(other_idx) == self.num_classes - 1:
                        pop_idx = np.random.randint(low=0, high=self.num_classes - 1)
                        other_idx = torch.cat((other_idx[0:pop_idx],other_idx[pop_idx+1:]), dim=0)
                    other_idxs[i] = other_idx

                other_angle_distances = torch.gather(angle_distances, dim=-1, index=other_idxs)

                min_other_angle = torch.min(other_angle_distances, dim=-1, keepdim=True)

                min_other_angle_distance =  min_other_angle.values
                min_other_angle_idxs =  min_other_angle.indices

                other_other_idxs = torch.empty(len(labels), self.num_classes - 3, dtype=torch.int64).cuda()

                for i in range(len(labels)):
                    other_other_idxs[i] = torch.cat((other_idxs[i][0: min_other_angle_idxs[i]], other_idxs[i][min_other_angle_idxs[i] + 1:]), dim=0)

                other_other_angle_distances = torch.gather(angle_distances, dim=-1, index=other_other_idxs)

                top_k = np.maximum(int((self.num_classes - 3) * self.top_rate), 1)

                other_other_topk = torch.sort(other_other_angle_distances, dim=-1).values[:,:top_k]

                other_other_average_distances = other_other_topk.mean(dim=-1)

                assert torch.min(other_other_average_distances.squeeze() - min_other_angle_distance.squeeze()) >= 0

                if self.if_aum == 1:

                    weights = torch.sigmoid(
                        self.alpha * (min_other_angle_distance.squeeze() - labeled_angle_distance.squeeze())) \
                              + torch.exp(
                        self.beta * (min_other_angle_distance.squeeze() - other_other_average_distances.squeeze()))

                    weights = weights / 2.

                else:
                    weights = torch.exp(self.beta * (min_other_angle_distance.squeeze() - other_other_average_distances.squeeze()))
                #logits = F.normalize(logits, dim=-1)

                revise_distance = torch.sort(other_angle_distances, dim=-1).values[:, 1]

                revise_idxs = torch.where(revise_distance < labeled_angle_distance.squeeze())[0]

                f_weights = torch.ones_like(weights)
                f_weights[revise_idxs] = weights[revise_idxs]

                final_weights = f_weights.squeeze().unsqueeze(dim=-1)

            out_weights = final_weights.detach()

            return logits, out_weights


