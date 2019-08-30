
data_root = "data"

dataset_mean = (0.5, 0.5, 0.5)
dataset_std = (0.5, 0.5, 0.5)

batch_size = 50
finetune_flag = 1

#cw_flag = 1
#multi_loss_flag = 0
#loss_recon_weight = 1
#flag3d = 1
holistic_size = 200
#image_size = 64

# params for source dataset

src_dataset = "BP4D_train1"
src_model_trained = True

tgt_dataset = "BP4D_test1"
tgt_model_trained = True


# params for training network
num_gpu = 2
num_epochs_pre = 5

log_step_pre = 100
eval_step_pre = 100
save_step_pre = 1
num_epochs = 40
log_step = 100
save_step = 1
manual_seed = None

c_learning_rate = 1e-3

