import params
import os
from core import eval_tgt
from models import PatchEncoder, PatchAttention, PatchClassifier
from utils import get_data_loader, init_model

aus = [1,2,4,6,7,10,12,14,15,17,23,24]

if __name__ == '__main__':

    src_name = params.src_dataset
    tgt_name = params.tgt_dataset

    folder = src_name + 'To' + tgt_name

    for i in range(0,12):
        for s in range(1, 6): #epoch
            for iter in range(500,500,2000):

                path_snapshot = 'snapshots_AU' + str(aus[i]) + '/Epoch-' + str(s) + '/'
                path_output = 'outputs_AU' + str(aus[i]) + '/Epoch-' + str(s) + '/'
                src_encoder_restore = path_snapshot + '/source-encoder-' + str(iter) + '.pt'
                outputs_root = path_output + '/s' + str(s)
                if not os.path.exists(outputs_root):
                    os.makedirs(outputs_root)

                so_name = 'iter' + str(iter)
                src_data_loader = get_data_loader(params.src_dataset)
                src_data_loader_eval = get_data_loader(params.src_dataset, train=False)

                src_encoder = init_model(net=PatchEncoder(),
                                          restore=src_encoder_restore)

                attn_list = []
                classifier_list = []
                for j in range(0, 1):
                    src_attention_restore = path_snapshot + '/source-attn-' + str(j) + '-' + str(iter) + '.pt'
                    src_attention = init_model(net=PatchAttention(),
                                                restore=src_attention_restore)
                    attn_list.append(src_attention)
                    src_classifier_restore = path_snapshot + '/source-classifier-' + str(j) + '-' + str(iter) + '.pt'

                    src_classifier = init_model(net=PatchClassifier(),
                                                 restore=src_classifier_restore)
                    classifier_list.append(src_classifier)

                model_root_src = path_snapshot

                tgt_data_loader = get_data_loader(params.tgt_dataset)
                tgt_data_loader_eval = get_data_loader(params.tgt_dataset, train=False)

                eval_tgt(src_encoder, attn_list, classifier_list, tgt_data_loader_eval, path_output + '/' + so_name, i)
