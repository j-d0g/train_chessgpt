# ChessGPT-2 1.1B
out_dir = "out-stockfish-xxl-36"
eval_interval = 8000
eval_iters = 100
log_interval = 50
always_save_checkpoint = True

wandb_log = True
wandb_project = "chess-gpt"
dataset = "hf_dataset_stockfish"
wandb_run_name = "stockfish-xxl-36"
gradient_accumulation_steps = 4
batch_size = 28
block_size = 1023  # context of up to 1023 tokens (because dataset block size is 1024)

# 16-layer GPT model
n_layer = 36
n_head = 20
n_embd = 1600
dropout = 0.0

learning_rate = 3e-4
max_iters = 600000
lr_decay_iters = max_iters  # make equal to max_iters usually
min_lr = 3e-5  # learning_rate / 10 usually
beta2 = 0.95  # make a bit bigger because number of tokens per iter is small

warmup_iters = 2000  # not super necessary potentially
compile = False

# 30 doesn't work
# 28 gets 59.5% mfu