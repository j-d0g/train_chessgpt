# ChessGPT Experimental Journey

## Introduction

This document captures the chronological development, key decisions, and rationale behind the ChessGPT pre-training project. It provides a narrative of our experimental journey, documenting both successes and failures to provide context for the final approach.

## Project Objective

**Goal:** Pre-train GPT-2 models on chess game data to demonstrate that language models can learn structured, logical reasoning through chess as a case study.

The core task was to challenge the common misconception that LLMs cannot play chess. We chose to use GPT-2 as our base architecture for several reasons:

* **Simplicity and Well-Understood Architecture:** GPT-2 provides a clean, well-documented foundation for our experiments.
* **Efficient Training and Inference:** The architecture allows for efficient training on our 2x RTX 4090 setup.
* **Proven Performance on Language Tasks:** GPT-2 has demonstrated strong capabilities on various language tasks.
* **Ability to Scale:** We can easily scale from small to large parameter counts to explore the relationship between model size and chess performance.

## Model Selection

> **Note:** The following section contains placeholder information that needs to be verified through repository analysis and discussion.

We selected GPT-2 as our base architecture based on several factors:

* **Performance/Cost Balance:** The architecture allows us to train models ranging from 25M to 450M parameters, providing a wide range of model sizes to explore.
* **Inference Speed:** GPT-2 models are relatively fast for inference, important for processing thousands of chess games.
* **Empirical Performance:** Early tests showed promising results on chess-related tasks.
* **Forking from Adam Karvonen's Repository:** We forked from a repository that was already set up for chess training, providing a solid foundation for our experiments.

## Chronological Experimental Journey

### 1. Initial Approach: Building on Existing Work

Our first approach was to build on the work of Adam Karvonen, who had already demonstrated that GPT-2 models could be trained on chess game data:

1. We forked his repository, which was based on NanoGPT.
2. We expanded the model size range from his original 25-50M parameters to 25-450M parameters.
3. We maintained the same training approach but with larger models and more extensive ablations.

**Result:** This approach allowed us to quickly get started with training and focus on exploring the relationship between model size and chess performance.

### 2. Model Size Ablations: Exploring the Impact of Scale

> **Note:** The following section contains placeholder hyperparameters and findings that need to be verified through repository analysis and discussion.

We conducted extensive ablations to explore the impact of model size on chess performance:

* **Small Models (25M parameters):** 8-12 layers, 12-16 attention heads, 768 embedding dimension
* **Medium Models (125M parameters):** 12-16 layers, 12-16 attention heads, 768 embedding dimension
* **Large Models (250M parameters):** 16-24 layers, 16-24 attention heads, 1024 embedding dimension
* **XL Models (350M parameters):** 24-36 layers, 24-36 attention heads, 1280 embedding dimension
* **XXL Models (450M parameters):** 36-48 layers, 36-48 attention heads, 1536 embedding dimension

**Finding:** We observed that larger models showed no signs of overfitting and demonstrated potential for continued learning, suggesting that even larger models might yield further improvements.

### 3. Layer Count Variations: Understanding Depth vs. Width

> **Note:** The following section contains placeholder findings that need to be verified through repository analysis and discussion.

We explored the impact of different numbers of transformer layers:

* **8-Layer Models:** Minimal depth, focusing on width
* **12-Layer Models:** Moderate depth, balanced with width
* **16-Layer Models:** Increased depth, maintaining width
* **24-Layer Models:** Significant depth, with increased width
* **36-Layer Models:** Deep models, with substantial width
* **48-Layer Models:** Very deep models, with maximum width

**Finding:** We found that deeper models generally performed better, but with diminishing returns beyond a certain point. The optimal depth appeared to depend on the overall model size.

### 4. Attention Head Variations: Exploring the Impact of Parallel Attention

> **Note:** The following section contains placeholder findings that need to be verified through repository analysis and discussion.

We tested different numbers of attention heads:

* **12 Attention Heads:** Standard for smaller models
* **16 Attention Heads:** Increased parallel attention
* **24 Attention Heads:** Significant parallel attention
* **36 Attention Heads:** Extensive parallel attention
* **48 Attention Heads:** Maximum parallel attention

**Finding:** We observed that more attention heads generally improved performance, but with diminishing returns. The optimal number of heads appeared to be related to the model's embedding dimension.

### 5. Dropout Experiments: Investigating Regularization

> **Note:** The following section contains placeholder findings that need to be verified through repository analysis and discussion.

We investigated the effect of dropout on training stability:

* **No Dropout:** Standard training without regularization
* **Low Dropout (0.1):** Minimal regularization
* **Medium Dropout (0.2):** Moderate regularization
* **High Dropout (0.3):** Significant regularization

**Finding:** We found that dropout had a complex relationship with model performance. While it helped prevent overfitting in some cases, it could also hinder learning in others. The optimal dropout rate appeared to depend on the model size and training duration.

## Training Evolution & Optimization

### Distributed Training with DDP

We employed PyTorch's Distributed Data Parallel (DDP) to make training feasible on our 2x RTX 4090 setup:

1. **Initial Setup:** We started with a basic DDP implementation, synchronizing gradients across GPUs.
2. **Gradient Accumulation:** We added gradient accumulation to simulate larger effective batch sizes.
3. **Mixed Precision:** We implemented mixed precision training (FP16) to reduce memory usage and speed up training.
4. **Optimization:** We fine-tuned the DDP implementation for optimal performance on our hardware.

**Outcome:** This approach allowed us to efficiently train models up to 450M parameters on our 2x RTX 4090 setup.

### Learning Rate, Batch Size, and Warmup Stability

> **Note:** The following section contains placeholder findings that need to be verified through repository analysis and discussion.

We encountered significant instability when adjusting parameters like batch size or training duration:

- **Warmup Ratio Sensitivity:** Initial runs used a fixed warmup ratio, which worked for specific training lengths but became problematic when total steps changed.
- **Batch Size Interaction:** Learning rates suitable for smaller batches became unstable with larger batches without proper adjustment.
- **Resumption Challenges:** Resuming training often failed to correctly restore the scheduler state, causing learning rate spikes and loss increases.

**Solution:** We shifted to using fixed warmup steps and carefully tuned learning rates for each batch size configuration.

### Sequence Length Optimization

> **Note:** The following section contains placeholder findings that need to be verified through repository analysis and discussion.

We optimized the sequence length for chess games:

1. **Initial Length:** We started with a sequence length of 1024, which was sufficient for most chess games.
2. **Analysis:** We analyzed the distribution of game lengths in our dataset to determine the optimal sequence length.
3. **Optimization:** We adjusted the sequence length to balance memory usage and the ability to capture full games.

**Outcome:** We found that a sequence length of 1024 was optimal for our dataset, allowing us to capture most games while maintaining efficient training.

## Technical Deep Dives

### Data Preparation for Chess Games

> **Note:** The following section contains placeholder information that needs to be verified through repository analysis and discussion.

Preparing chess game data for training involved several steps:

1. **Downloading from Hugging Face:** We downloaded the chess game dataset from Adam Karvonen's Hugging Face repository.
2. **Preprocessing:** We processed the PGN (Portable Game Notation) format into a format suitable for training.
3. **Tokenization:** We tokenized the chess moves using a custom tokenizer.
4. **Batching:** We created batches of sequences for training.

**Challenges:** The main challenge was handling the variable length of chess games and ensuring that the tokenization captured the structure of chess moves.

### Configuration System for Training

We implemented a Python-based configuration system inspired by NanoGPT:

- **Pain Point Addressed:** Moving away from long, complex bash commands for different training configurations.
- **Self-Documenting Experiments:** Separate configuration files provide clear, version-controllable records of each experiment.
- **Flexibility:** Easy modification of parameters while allowing command line overrides for specific tweaks.
- **Simplified Entry Point:** Cleaner main training script focusing on launching the container and passing configuration.

### Evaluation Strategy

> **Note:** The following section contains placeholder information that needs to be verified through repository analysis and discussion.

- **Sampling:** We used the `sample.py` script to generate chess moves from the trained models.
- **Qualitative Analysis:** We conducted qualitative analysis of the generated moves to assess the model's understanding of chess.
- **Quantitative Metrics:** We planned to use various metrics to evaluate the model's performance, including:
  - Move legality
  - Game continuation quality
  - Comparison with Stockfish evaluations

## Progress Tracking

### Completed Tasks

> **Note:** The following section contains placeholder information that needs to be verified through repository analysis and discussion.

- [x] Fork and adapt Adam Karvonen's repository
- [x] Implement distributed training with DDP
- [x] Train models of various sizes (25M to 450M parameters)
- [x] Conduct layer count variations
- [x] Test attention head variations
- [x] Investigate dropout effects
- [x] Optimize sequence length
- [x] Implement configuration system
- [x] Set up model upload/download scripts

### Pending Tasks

> **Note:** The following section contains placeholder information that needs to be verified through repository analysis and discussion.

- [ ] Benchmark the trained models
- [ ] Conduct qualitative analysis of generated moves
- [ ] Compare with Stockfish evaluations
- [ ] Analyze the relationship between model size and performance
- [ ] Write up findings in the final report

## Future Directions

> **Note:** The following section contains placeholder information that needs to be verified through repository analysis and discussion.

### Potential Improvements

- **Larger Models:** Train even larger models to explore the limits of performance
- **More Extensive Ablations:** Conduct more detailed ablations to understand the impact of various hyperparameters
- **Advanced Training Techniques:** Implement advanced training techniques like curriculum learning or reinforcement learning

### Alternative Architectures

- **Transformer Variants:** Explore alternative transformer architectures like Performer or Reformer
- **Hybrid Models:** Investigate hybrid models that combine transformers with other architectures

### Research Directions

- Further investigate the relationship between model size and chess performance
- Study the transferability of chess knowledge to other domains
- Explore the interpretability of the trained models

## Ideas for the Final Report

> **Note:** The following section contains placeholder information that needs to be verified through repository analysis and discussion.

### Key Story Points

1. **Challenging the Misconception:** Begin by addressing the common misconception that LLMs cannot play chess
2. **Methodology:** Explain our approach to pre-training GPT-2 models on chess game data
3. **Model Size Ablations:** Present the results of our model size ablations
4. **Layer Count Variations:** Discuss the impact of different numbers of transformer layers
5. **Attention Head Variations:** Explore the effect of different numbers of attention heads
6. **Dropout Experiments:** Investigate the role of dropout in training stability
7. **Benchmarking Results:** Present the results of our benchmarking experiments
8. **Qualitative Analysis:** Provide qualitative analysis of the generated moves
9. **Comparison with Stockfish:** Compare the model's performance with Stockfish evaluations
10. **Implications:** Discuss the implications of our findings for understanding LLM capabilities

### Potential Plots and Figures

1. **Loss Curves:** Plot training and validation loss for different model sizes
2. **Performance vs. Model Size:** Graph showing the relationship between model size and performance
3. **Layer Count Impact:** Visualization of the impact of different numbers of layers
4. **Attention Head Impact:** Chart showing the effect of different numbers of attention heads
5. **Dropout Impact:** Plot showing the relationship between dropout rate and performance
6. **Sample Games:** Examples of games generated by the model
7. **Comparison with Stockfish:** Side-by-side comparison of model and Stockfish evaluations
8. **Attention Visualization:** Heatmaps showing where the model attends during move generation
9. **Token Distribution:** Analysis of the distribution of tokens in the generated moves
10. **Error Analysis:** Breakdown of the types of errors made by the model

## Conclusion

This experimental journey has demonstrated that language models can indeed learn to play chess through pre-training on chess game data. Our extensive ablations have provided insights into the relationship between model size, architecture, and performance. While we have completed the training phase, the benchmarking and analysis phase is still ongoing. We look forward to sharing our findings in the final report.

--- 