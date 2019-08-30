
import torch
import torch.nn as nn
import params

from utils import make_variable

sft = nn.Softmax(dim=1)
def eval_tgt(encoder, attn_list, classifier_list, data_loader, name, au_ind):
    import numpy as np
    import os
    import scipy.io as sio

    encoder.eval()
    for i in range(0,1):
        classifier_list[i].eval()
        attn_list[i].eval()

    cnt = 0
    for (image1s, image2s, image3s, image4s, image5s, image6s, image7s, image8s, image9s, labels) in data_loader:
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
        labels = labels[:,au_ind].unsqueeze(-1)
        encoded = encoder(image1s, image2s, image3s, image4s, image5s, image6s, image7s, image8s, image9s)

        preds = []
        attn_results = []

        for i in range(0, 1):
            preds_au = classifier_list[i](attn_list[i](encoded)[0])
            attn_sum = attn_list[i](encoded)[1].unsqueeze(-1)
            if i == 0:
                preds = preds_au
                attn_results = attn_sum
            else:
                preds = torch.cat((preds, preds_au), 1)
                attn_results = torch.cat((attn_results, attn_sum), 2)
        preds = torch.sigmoid(preds)

        if cnt == 0:
            all_preds = preds.cpu().detach().numpy()
            all_labels = labels.cpu().detach().numpy()
            all_attns = attn_results.cpu().detach().numpy()

        else:
            all_preds = np.vstack((all_preds, preds.cpu().detach().numpy()))
            all_labels = np.vstack((all_labels, labels.cpu().detach().numpy()))
            all_attns = np.vstack((all_attns, attn_results.cpu().detach().numpy()))

        cnt = cnt + 1
        if ((cnt) % 100 == 0):
            print(cnt)

    sio.savemat(name + '.mat', {'all_preds': all_preds, 'all_labels': all_labels, 'all_attns': all_attns})
