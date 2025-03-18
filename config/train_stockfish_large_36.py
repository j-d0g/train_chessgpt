# ChessGPT-2 450M
out_dir = "out-stockfish-large-36"
eval_interval = 4000
eval_iters = 100
log_interval = 50
always_save_checkpoint = True

wandb_log = True
wandb_project = "chess-gpt"
dataset = "hf_dataset_stockfish"
wandb_run_name = "stockfish-large-36"

wandb_id = "eeab8bdt"
init_from = "resume" if wandb_id else "scratch"

gradient_accumulation_steps = 4
batch_size = 32
block_size = 1023  # context of up to 1023 tokens (because dataset block size is 1024)

# 16-layer GPT model
n_layer = 36
n_head = 16
n_embd = 1024
dropout = 0.0

learning_rate = 3e-4
max_iters = 600000
lr_decay_iters = max_iters  # make equal to max_iters usually
min_lr = 3e-5  # learning_rate / 10 usually
beta2 = 0.95  # make a bit bigger because number of tokens per iter is small

warmup_iters = 2000  # not super necessary potentially
compile = False

# 48 gets 54% mfu at 1800ms
# 52 gets 55% mfu at 1900ms