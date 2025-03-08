# train a medium-sized Chess GPT model on RTX4090
# optimizer for GPU training

out_dir = 'out-chess-gpu'
eval_interval = 2000
eval_iters = 100
log_interval = 10

# save checkpoints
always_save_checkpoint = True

wandb_log = True  # set to False if you don't want to use wandb
wandb_project = 'chess-gpt'
wandb_run_name = 'chessgpt-rtx4090'

dataset = 'lichess_hf_dataset'
gradient_accumulation_steps = 1
batch_size = 32  # with RTX4090 we can use larger batch sizes
block_size = 512  # context length

# medium-sized model parameters
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.1

learning_rate = 5e-4
max_iters = 20000
lr_decay_iters = 20000
min_lr = 5e-5
beta2 = 0.95

warmup_iters = 500

# GPU-specific settings
device = 'cuda'
compile = False
dtype = 'bfloat16'