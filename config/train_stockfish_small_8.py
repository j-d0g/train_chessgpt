# ChessGPT-2 25M
out_dir = "out-stockfish_small_8"
eval_interval = 4000
eval_iters = 100
log_interval = 10

always_save_checkpoint = True

wandb_log = True  # Keep wandb logging enabled
wandb_project = "chess-gpt"
dataset = "hf_dataset_stockfish"
wandb_run_name = "stockfish-small-8"

gradient_accumulation_steps = 1
batch_size = 100
block_size = 1023  # context of up to 1023 tokens (because dataset block size is 1024)

# baby GPT model :)
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.0

learning_rate = 3e-4
max_iters = 600000
lr_decay_iters = max_iters  # make equal to max_iters usually
min_lr = 3e-5  # learning_rate / 10 usually
beta2 = 0.95  # make a bit bigger because number of tokens per iter is small

warmup_iters = 500  # not super necessary potentially
compile = False

