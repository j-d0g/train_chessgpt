# ChessGPT-2 300M
out_dir = "out-stockfish-large-24"
eval_interval = 8000
eval_iters = 100
log_interval = 50
always_save_checkpoint = True

wandb_log = True
wandb_project = "chess-gpt"
dataset = "hf_dataset_stockfish"
wandb_run_name = "stockfish-large-24"

wandb_id = "h4fde5ls"
init_from = "resume" if wandb_id else "scratch"

gradient_accumulation_steps = 2
batch_size = 64
block_size = 1023  # context of up to 1023 tokens (because dataset block size is 1024)

# 16-layer GPT model
n_layer = 24
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

# 64 gets 56% mfu 1500ms (best)
# 32 gets 54% mfu 1600ms