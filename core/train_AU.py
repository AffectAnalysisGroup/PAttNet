import torch.nn as nn
import torch.optim as optim

import params
from utils import make_variable, save_model
import torch._utils
from torch.autograd import Variable
import os

def weighted_binary_cross_entropy_with_logits(logits, targets, pos_weight, weight=None, size_average=True, reduce=True):
    if not (targets.size() == logits.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(targets.size(), logits.size()))

    max_val = (-logits).clamp(min=0)
    log_weight = 1 + (pos_weight - 1) * targets
    loss = (1 - targets) * logits + log_weight * (((-max_val).exp() + (-logits - max_val).exp()).log() + max_val)

    if weight is not None:
        loss = loss * weight

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()

class WeightedBCEWithLogitsLoss(torch.nn.Module):
    def __init__(self, pos_weight, weight=None, size_average=True, reduce=True):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, input, target):
        pos_weight = Variable(self.pos_weight) if not isinstance(self.pos_weight, Variable) else self.pos_weight
        if self.weight is not None:
            weight = Variable(self.weight) if not isinstance(self.weight, Variable) else self.weight
            return weighted_binary_cross_entropy_with_logits(input, target,
                                                             pos_weight,
                                                             weight=weight,
                                                             size_average=self.size_average,
                                                             reduce=self.reduce)
        else:
            return weighted_binary_cross_entropy_with_logits(input, target,
                                                             pos_weight,
                                                             weight=None,
                                                             size_average=self.size_average,
                                                             reduce=self.reduce)

if params.src_dataset == 'BP4D41_train1':
    cw = torch.cuda.FloatTensor([3.2349, 4.6748, 3.892,	1.0638,	0.8957,	0.6564,	0.7735,	1.0003,	4.5920,	1.8030,	4.9155, 5.3944])

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2



def train_src(encoder, attn_list, classifier_list, data_loader, model_root, au_ind):
    ####################
    # 1. setup network #
    ####################


    encoder.train()
    for i in range(0,1):
        classifier_list[i].train()
        attn_list[i].train()

    cw_au = cw[au_ind]

    parameter_list = list(encoder.parameters())
    for i in range(0,1):
        parameter_list = parameter_list + list(classifier_list[i].parameters()) + list(attn_list[i].parameters())

    optimizer = optim.SGD(parameter_list, lr=params.c_learning_rate, momentum=0.9)
    criterion = WeightedBCEWithLogitsLoss(pos_weight=cw_au)

    ####################
    # 2. train network #
    ####################

    for epoch in range(params.num_epochs_pre):
        if not os.path.exists(model_root + "/Epoch-{}/".format(epoch + 1)):
            os.makedirs(model_root + "/Epoch-{}/".format(epoch + 1))
            
        for step, (image1s, image2s, image3s, image4s, image5s, image6s, image7s, image8s, image9s, labels) in enumerate(data_loader):
            # make images and labels variable
            image1s = make_variable(image1s)
            image2s = make_variable(image2s)
            image3s = make_variable(image3s)
            image4s = make_variable(image4s)
            image5s = make_variable(image5s)
            image6s = make_variable(image6s)
            image7s = make_variable(image7s)
            image8s = make_variable(image8s)
            image9s = make_variable(image9s)
            labels = make_variable(labels.squeeze_()).float()

            labels[labels == -1] = 0

            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for critic
            encoded = encoder(image1s, image2s, image3s, image4s, image5s, image6s, image7s, image8s, image9s)

            preds = []
            for i in range(0,1):
                preds_au = classifier_list[i](attn_list[i](encoded)[0])
                if i == 0:
                    preds = preds_au
                else:
                    preds = torch.cat((preds, preds_au), 1)

            loss = criterion(preds, labels[:,au_ind].unsqueeze(dim=-1))
            loss.backward()

            optimizer.step()

            # print step info
            if (step + 1) % 100 == 0:
                print("Epoch [{}/{}] Step [{}/{}]: loss={}"
                      .format(epoch + 1,
                              params.num_epochs_pre,
                              step + 1,
                              len(data_loader),
                              loss.item()))

            if (step + 1) % 500 == 0:

                save_model(encoder, "source-encoder-{}.pt".format(step + 1),
                           model_root + "/Epoch-{}/".format(epoch + 1))

                for i in range(0, 1):
                    save_attn_name = 'source-attn-' + str(i) + '-{}.pt'.format(step + 1)
                    save_model(attn_list[i], save_attn_name, model_root + "/Epoch-{}/".format(epoch + 1))
                    save_classifier_name = 'source-classifier-' + str(i) + '-{}.pt'.format(step + 1)
                    save_model(classifier_list[i], save_classifier_name, model_root + "/Epoch-{}/".format(epoch + 1))

    return encoder, attn_list, classifier_list

