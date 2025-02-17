import time

#out_dir = 'out-owt-gpt2mini'
out_dir = 'out-jam-cgpt'
eval_interval = 10
eval_iters = 80
wandb_log = True
wandb_project = 'jam-cgpt-fine-tuning-Precise-Axis'
wandb_run_name = 'jam-cgpt-fine-tuning-model_human-precise_epoch3'

dataset = 'Precise'
init_from = 'resume'

# only save checkpoints if the validation loss improves
always_save_checkpoint = True 

# jam-cgpt 170k dataset has 37,399,419 tokens

# model iters
# 38m parameters model has 757,000 iters
# 110m parameters model has 762,000 iters
# 350m parameters model has 272,000 iters

block_size = 256

batch_size = 4 #16
gradient_accumulation_steps = 32


#max_iters = 272000 + 1150 * 3
# max_iters = 308900 + 275 * 3 + 5 * 5 #Condensing
max_iters = 308900 + 275 * 3 + 5 * 3
# finetune at constant LR
learning_rate = 3e-5
decay_lr = False
