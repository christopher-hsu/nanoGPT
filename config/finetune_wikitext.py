import time

out_dir = 'out-wikitext-ft'
eval_interval = 5
eval_iters = 40
wandb_log = False # feel free to turn on
wandb_project = 'wikitext'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'wikitext'
init_from = 'gpt2-medium' # gpt2-xl this is the largest GPT-2 model

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
# batch_size = 1
# gradient_accumulation_steps = 1

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 1
block_size = 1024
gradient_accumulation_steps = 32

max_iters = 20

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False