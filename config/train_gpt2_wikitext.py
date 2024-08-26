# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'wikitext'
wandb_run_name='gpt2-15k-8bs-8gs-12l-12h-768e'
# wandb_run_name='gpt2-15k-4bs-8gs-24l-16h-1024e'

out_dir = 'out-wikitext-15k-8bs-8gs-12l-12h-768e'
# out_dir = 'out-wikitext-15k-4bs-8gs-24l-16h-1024e'
dataset = 'wikitext'
# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 8
block_size = 1024
gradient_accumulation_steps = 8

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 #max learning rate
max_iters = 20000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 20000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla


# eval stuff
eval_interval = 100
eval_iters = 512
log_interval = 10
