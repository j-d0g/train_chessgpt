# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-chess-mps'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 50
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # set to True if you want to log to wandb
wandb_project = 'chess-gpt'

dataset = 'lichess_hf_dataset'
gradient_accumulation_steps = 4
batch_size = 1
block_size = 128 # reduced context size for faster training

# lighter model for Macbook training
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0

learning_rate = 1e-3
max_iters = 2000
lr_decay_iters = 2000
min_lr = 1e-4
beta2 = 0.95

warmup_iters = 100

# Mac-specific settings
device = 'mps'  # use MPS (Metal Performance Shaders)
compile = False # do not torch compile the model
