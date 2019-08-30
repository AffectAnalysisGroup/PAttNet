import params
import os
from core import  train_src
from models import PatchEncoder, PatchClassifier, PatchAttention
from utils import get_data_loader, init_model
import torch

torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)

aus = [1,2,4,6,7,10,12,14,15,17,23,24]

if __name__ == '__main__':

    src_name = params.src_dataset
    tgt_name = params.tgt_dataset

    folder = src_name + 'To' + tgt_name

    s = 10

    for i in range(0,12):
        path_snapshot = 'snapshots_AU' + str(aus[i]) + '/'
        path_output = 'outputs_AU' + str(aus[i]) + '/'

        src_encoder_restore = path_snapshot + '/source-encoder-' + str(s) + '.pt'
        src_attention_restore = path_snapshot + '/source-attn-' + str(s) + '.pt'
        src_classifier_restore = path_snapshot + '/source-classifier-' + str(s) + '.pt'

        src_data_loader = get_data_loader(params.src_dataset)
        src_data_loader_eval = get_data_loader(params.src_dataset, train=False)

        src_encoder = init_model(net=PatchEncoder(),
                                  restore=src_encoder_restore)
        attn_list = []
        classifier_list = []
        for j in range(0,1):
            src_attention = init_model(net=PatchAttention(),
                                  restore=src_attention_restore)
            attn_list.append(src_attention)
            src_classifier = init_model(net=PatchClassifier(),
                                     restore=src_classifier_restore)
            classifier_list.append(src_classifier)
        model_root_src = path_snapshot

        if not os.path.exists(model_root_src):
            os.makedirs(model_root_src)
        print(i)
        if not (src_encoder.restored and
                params.src_model_trained):
            src_encoder, attn_list, classifier_list = train_src(
                src_encoder, attn_list, classifier_list, src_data_loader, model_root_src, i)

