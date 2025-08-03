# ChessGPT Training: Systematic Scaling Studies in Chess Language Models

## 📋 Table of Contents

1. [🎯 Motivation and Context](#1--motivation-and-context)
   - 1.1 [Research Context](#11-research-context)
   - 1.2 [Why Chess and Next-Token Prediction?](#12-why-chess-and-next-token-prediction)
2. [📚 Research Foundation](#2--research-foundation)
   - 2.1 [Building on Prior Work](#21-building-on-prior-work)
   - 2.2 [NanoGPT Framework Choice](#22-nanogpt-framework-choice)
3. [🔬 Experimental Design](#3--experimental-design)
   - 3.1 [Dataset and Training Setup](#31-dataset-and-training-setup)
     - 3.1.1 [Dataset Overview](#311-dataset-overview)
     - 3.1.2 [Data Processing and Block Structure](#312-data-processing-and-block-structure)
     - 3.1.3 [Strategic Design Rationale](#313-strategic-design-rationale)
     - 3.1.4 [Vocabulary Efficiency Analysis](#314-vocabulary-efficiency-analysis)
   - 3.2 [Model Architecture Variants](#32-model-architecture-variants)
   - 3.3 [Training Infrastructure](#33-training-infrastructure)
4. [📊 Training Results](#4--training-results)
   - 4.1 [Baseline Replication](#41-baseline-replication)
   - 4.2 [Scaling Observations](#42-scaling-observations)
     - 4.2.1 [Performance Summary](#421-performance-summary)
     - 4.2.2 [Depth vs. Width Analysis](#422-depth-vs-width-analysis)
     - 4.2.3 [Training Efficiency Analysis](#423-training-efficiency-analysis)
     - 4.2.4 [Training Dynamics and Convergence](#424-training-dynamics-and-convergence)
   - 4.3 [Training Dynamics](#43-training-dynamics)
5. [⚙️ Technical Implementation](#5--technical-implementation)
6. [🚀 Reflection and Next Steps](#6--reflection-and-next-steps)
   - 6.1 [What This Training Study Accomplished](#61-what-this-training-study-accomplished)
   - 6.2 [Questions Raised for Future Investigation](#62-questions-raised-for-future-investigation)
   - 6.3 [Connection to Broader Research Program](#63-connection-to-broader-research-program)

---

## 1. 🎯 Motivation and Context

This work sits within a broader research program investigating whether large language models develop internal world models when trained on structured domains. The specific focus here is on the pre-training phase: systematically training transformer models on chess game data to understand how architectural choices and scale affect the learning of chess representations.

Initial motivations were to verify preliminary evidence that frontier LLMs such as GPT-3.5's could play chess beyond shallow distribution matching by replicating a similar system that would give me access to the pre-training data. Without this, any number of games produced by the model, no matter how improbable, would have insufficient evidence of generalisation beyond pre-training distribution matching.

However, given recent contributions to the field by Neel Nanda and Kenneth Li on Othello-GPT, we go beyond such proofs to mechanistically investigate and verify the natural questions that arise from apparent emergence. One of such is that, in order to play legal moves beyond its training set of games, it must be able to track the board-state. By probing our model's internal weight activations on the task of board-state tracking, we can confirm whether this is true.

The pre-requisite to these experiments is to first pre-train our own GPT models on sequential chess data. We train multiple ablations using GPT-2-inspired scaling laws in order to observe the emergent properties and effects of parameters, length, and width on how LLMs internalise features in such a logical problem set.

### 1.1 Research Context

> This study builds directly on two key papers: [Li et al. (2023)](https://arxiv.org/abs/2210.13382) on Othello-GPT demonstrating emergent world models, and [Karvonen (2023)](https://adamkarvonen.github.io/machine_learning/2024/01/03/chess-world-models.html) extending Chess-GPT to real chess data.

The foundational work by Li et al. (2023) on Othello-GPT demonstrated that language models could learn internal board representations when trained on game sequences. However, their work focused on synthetic data and a simpler game. Adam Karvonen extended this to chess using real game data, showing that GPT models could learn to play chess through next-token prediction and develop internal board state representations detectable through linear probing.

This training study investigates:
- How different architectural configurations (depth vs. width) affect training dynamics
- Whether scaling trends observed in language modeling hold for structured domains like chess  
- How training progresses across different model sizes and what representations emerge

### 1.2 Why Chess and Next-Token Prediction?

Chess provides an ideal testbed for studying world model emergence because:
- **Deterministic rules**: Clear ground truth for evaluating internal representations
- **Rich structure**: Complex enough to require abstract reasoning beyond pattern matching  
- **Data availability**: Large datasets of real games from platforms like Lichess
- **Interpretability**: Board states and moves can be directly probed and validated

The next-token prediction approach is compelling because it mirrors how language models are trained, yet requires the model to implicitly learn chess rules and board state tracking to predict legal moves accurately.

## 2. 📚 Research Foundation

### 2.1 Building on Prior Work

> This study extends Adam Karvonen's Chess-GPT work², which demonstrated that 8-layer and 16-layer GPT models (25M and 50M parameters) can learn to play chess through next-token prediction, achieve 99%+ legal move rates, and develop internal board state representations detectable via linear probing.

Karvonen's work left several questions about scaling and architectural choices:
- How do smaller vs. larger / deeper vs. wider models compare in chess?
- Adam's training results demonstrated the model still had capacity to learn more with validation loss still linearly decreasing after 600,000 iterations; what training dynamics emerge as models grow larger / trained longer?
- How do different architectural configurations affect the development of internal representations for latent variables, i.e layers responsible for board-state tracking?
- What can this tell us about how LLMs learn world models from structured data representations?

### 2.2 NanoGPT Framework Choice

We use [Andrej Karpathy's NanoGPT](https://github.com/karpathy/nanoGPT)³ implementation for several practical reasons:
- **Clean, transparent codebase**: Easy to modify and understand training dynamics
- **Full experimental control**: Access to all model weights, training data, and hyperparameters
- **Efficient scaling**: Supports distributed training on multi-GPU setups
- **Reproducibility**: Well-documented, widely-used implementation with consistent results

**Key Adaptations from Standard NanoGPT:**
- **Vocabulary reduction**: From 50,304 tokens → 32 chess-specific tokens
- **Storage optimization**: int8 encoding vs int16 due to smaller vocabulary
- **Modified `get_batch()`**: Ensures training sequences start at game boundaries (`;` delimiter)
- **Block size**: 1024 characters optimized for chess game structure

**Architecture Comparison: ChessGPT vs Standard NanoGPT**

| Model Class | NanoGPT Standard | ChessGPT | Key Differences |
|-------------|------------------|----------|-----------------|
| **Small** | 12L × 12H × 768E (124M) | 8-36L × 8H × 512E (25-114M) | Deeper, narrower, vocab-efficient |
| **Medium** | 24L × 16H × 1024E (350M) | 12-16L × 12H × 768E (86-114M) | Shallower but compute-dense |
| **Large** | 36L × 20H × 1280E (774M) | 16L × 16H × 1024E (202M) | Dramatically more efficient |

**Architectural Rationale:** The vocabulary efficiency (99.94% reduction) enables allocating parameters to **depth and reasoning** rather than vocabulary representation. Adam's choice of 512 embedding dimension—narrower than GPT-2 small's 768—likely reflects this: with vocabulary overhead eliminated, the "saved" parameters can be invested in additional layers for enhanced chess reasoning capability.

Before beginning my pre-training, I followed the NN-Zero-To-Hero tutorials which walk through the foundational ML pre-requisites to understanding, diagnosing and pre-training your own GPT model from scratch. I initially conducted this using the shakespeare dataset before applying the techniques to chess game sequences.

## 3. 🔬 Experimental Design

### 3.1 Dataset and Training Setup

#### 3.1.1 Dataset Overview

**Dataset Source:** We use the same Stockfish self-play dataset from Adam Karvonen's work:
- **Source:** `stockfish_dataset_blocks.zip` from [adamkarvonen/chess_games](https://huggingface.co/datasets/adamkarvonen/chess_games)
- **Scale:** 16 million chess games
- **Game Structure:** Stockfish self-play with White (3200 ELO) vs. Black (1300-3200 ELO range)
- **Format:** PGN notation tokenized to **32-character vocabulary**

> **Key Design Choice:** The asymmetric ELO structure (strong White vs. variable Black) ensures the model learns predominantly winning patterns, crucial for language models that optimize for distribution replication rather than game outcomes.

#### 3.1.2 Data Processing and Block Structure

**Processing Pipeline:**
```
16M Games → Concatenated File → 1024-char Blocks → BTC Training Tensors
```

**Block Architecture:**
- **Block Size:** 1024 characters (≈90 chess moves)
- **Context Length:** 1023 tokens (reserving 1 for next-token prediction)
- **Game Delimiter:** `;` character marks game boundaries
- **Flexible Content:** Blocks contain 1-3 complete/partial games

**Boundary Handling Example:**
```
1.e4 e5...26.Rxf7+ Kg8 27.Qh6#;1.e4 e5 2. Nf3...
```

This design ensures efficient token utilization while enabling the model to learn game transition patterns through standard BTC (Batch-Time-Channel) autoregressive training.

#### 3.1.3 Strategic Design Rationale

**Human vs. Synthetic Data Insights:**  
Adam Karvonen's comparative experiments revealed a fascinating paradox:

| Data Type | Validation Loss | Chess Performance | Interpretation |
|-----------|----------------|-------------------|----------------|
| **Human Games** | ✅ Lower | ❌ Worse | More predictable patterns |
| **Stockfish Games** | ❌ Higher | ✅ Better | Complex but superior patterns |

**Implications:**
- **Stockfish data** contains complex patterns harder to model but leading to superior play
- **Human games** show predictable mistake patterns, making modeling easier but performance worse
- **Architectural alignment:** GPT patterns (attention, heuristics) may align better with human cognition than Stockfish's brute-force search

**Our Design Benefits:**
- **Diverse Opposition:** White's performance is fixed at 3200 ELO and learns against varied skill levels (1300-3200 ELO Black).
- **Robust Fundamentals:** Model internalizes material gain, tactical exploitation, positional advantage against various skill-levels.
- **Generalization:** Prevents overfitting to expert-only games that could make the model brittle against unexpected moves.

#### 3.1.4 Vocabulary Efficiency Analysis

**Parameter Allocation Advantage:**  
The 32-token chess vocabulary represents a **99.94% reduction** from nanoGPT's 50,304-token vocabulary, dramatically improving compute density:

| Model | Chess Params | Standard Vocab Equivalent | Efficiency Gain |
|-------|--------------|---------------------------|-----------------|
| **small-8** | 25.7M | 77.2M | **+200%** |
| **small-16** | 50.9M | 102.4M | **+101%** |
| **small-36** | 113.8M | 165.3M | **+45%** |
| **medium-12** | 85.8M | 163.0M | **+90%** |
| **large-16** | 202.5M | 305.5M | **+51%** |

**Key Insight:** Where standard models devote ~47% of parameters to vocabulary embeddings, our chess-specific approach allocates **~99.97% to reasoning**, making our models significantly more compute-dense than parameter counts suggest.

*Formula: Vocabulary overhead = 2 × |V| × |D| (embedding + output layers)*

### 3.2 Model Architecture Variants

We trained 7 model variants to systematically investigate scaling patterns:

| Model | Parameters | Layers | Heads | Embedding | Focus |
|-------|------------|--------|-------|-----------|-------|
| **small-8** | 25.7M | 8 | 8 | 512 | Baseline replication |
| **small-16** | 50.9M | 16 | 8 | 512 | Baseline replication |
| **small-24** | 76.1M | 24 | 8 | 512 | Deep narrow |
| **small-36** | 113.8M | 36 | 8 | 512 | Very deep narrow |
| **medium-12** | 85.8M | 12 | 12 | 768 | Balanced medium |
| **medium-16** | 114.1M | 16 | 12 | 768 | Deeper medium |
| **large-16** | 202.5M | 16 | 16 | 1024 | Wide architecture |

The first two variants (small-8, small-16) directly replicate Karvonen's best-performing architectures, providing validation baselines. The remaining variants explore different architectural trade-offs between depth and width.

### 3.3 Training Infrastructure

**Hardware and Optimization:** (verify GPU hours and set-ups used for each individual model via wandb later)
- **GPUs:** 1-2x RTX 4090 (24GB VRAM each)  
- **Framework:** PyTorch with Distributed Data Parallel (DDP)
- **Precision:** Mixed precision (FP16) for memory efficiency
- **Batch Sizes:** Adjusted per model size with gradient accumulation

**Training Hyperparameters:** (Needs verification of notes and why we chose things! Most are to keep consistent with Adam's set-up, but we can ask follow-up question: why did Adam or Neel Nanda choose these?)

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Optimizer** | AdamW (β₁=0.9, β₂=0.95, weight_decay=0.1) | Standard for transformers |
| **Learning Rate** | 3e-4 → 3e-5 (cosine decay) | Consistent across all models |
| **Training Steps** | 600,000 iterations | All models trained to completion |
| **Warmup** | 2000 steps | Consistent across all models 
| **Dropout** | 0.0 | Avoids redundancy that complicates interpretability* |

*Following Nanda's findings in OthelloGPT¹: dropout incentivizes redundancy where features spread across dimensions rather than being cleanly represented, making mechanistic analysis more difficult.

## 4. 📊 Training Results

### 4.1 Baseline Replication ✅ VERIFIED

Our first priority was validating the experimental setup by replicating Karvonen's baseline results:

**Verified Baseline Comparison:**

| Model | Parameters | Architecture | Our Val Loss | Adam's Val Loss* | Status |
|-------|------------|--------------|--------------|------------------|---------|
| **small-8** | 25.7M | 8L × 8H × 512E | **0.2944** | 0.2920 (W&B) / 0.2922 (ckpt) | ✅ **Replicated** |
| **small-16** | 50.9M | 16L × 8H × 512E | **0.2725** | 0.2712 (W&B) / 0.2716 (ckpt) | ✅ **Replicated** |

*Adam's validation losses verified from: [8-layer W&B run](https://wandb.ai/adam-karvonen/chess-gpt-batch/runs/ntimscy0/overview) and [16-layer W&B run](https://wandb.ai/adam-karvonen/chess-gpt-batch/runs/ygvhseer/overview), plus checkpoint analysis.

**Replication Status:** ✅ Successfully replicated baseline performance with validation loss differences of ~0.001-0.002, confirming experimental validity and providing reliable scaling study foundation.

### 4.2 Scaling Observations

#### 4.2.1 Performance Summary

**Model Performance Rankings:**  
| Rank | Model | Parameters | Best Val Loss | Key Insight |
|------|-------|------------|---------------|-------------|
| 🥇 | **large-16** | 202.4M | **0.257796*** | Best overall, checkpointing issue |
| 🥈 | **small-36** | 113.8M | **0.258288** | Exceptional depth efficiency |
| 🥉 | **medium-16** | 114.1M | **0.260812** | Similar params, worse than small-36 |
| 4th | **small-24** | 76.1M | **0.262813** | Strong depth scaling |
| 5th | **medium-12** | 85.8M | **0.265215** | Balanced architecture |
| 6th | **small-16** | 50.9M | **0.272486** | Replicated baseline |
| 7th | **small-8** | 25.7M | **0.294406** | Minimal baseline |

> ***Note:** large-16 achieved **0.25621** validation loss during training but checkpointing saved final iteration instead of best performance.

#### 4.2.2 Depth vs. Width Analysis

**Key Finding: Depth Outperforms Width**  

To directly compare depth and width, we focus on two sets of models with similar parameter counts:

**Comparison 1: High-Parameter Models**
| Model      | Layers | Hidden Size | Params   | Best Val Loss |
|------------|--------|-------------|----------|---------------|
| small-36   | 36     | 8           | 113.8M   | 0.258288      |
| medium-16  | 16     | 12          | 114.1M   | 0.260812      |
| large-16   | 16     | 16          | 202.4M   | 0.257796*     |

- **small-36** (deeper, thinner) outperforms **medium-16** (wider, shallower) at nearly identical parameter count, achieving a lower validation loss (0.2583 vs. 0.2608).
- **large-16** (widest, same depth as medium-16) achieves the best loss overall, but with nearly double the parameters.

**Comparison 2: Mid-Parameter Models**
| Model      | Layers | Hidden Size | Params   | Best Val Loss |
|------------|--------|-------------|----------|---------------|
| small-24   | 24     | 8           | 76.1M    | 0.262813      |
| medium-12  | 12     | 12          | 85.8M    | 0.265215      |

- **small-24** (deeper, thinner) outperforms **medium-12** (wider, shallower) despite having fewer parameters (0.2628 vs. 0.2652).

**Summary:**  
For both parameter ranges, deeper and thinner models (small-36, small-24) consistently outperform wider, shallower models (medium-16, medium-12) with similar or even fewer parameters. This highlights the parameter efficiency of depth for chess next-token prediction in this regime.


#### 4.2.3 Training Efficiency Analysis

**Computational Requirements:**

| Model | Architecture | Parameters | GPUs | Per-GPU Batch | Grad Accum | Effective Batch | Training Time | GPU Hours | Best Val Loss | Final Train Loss | Performance vs small-16 |
|-------|-------------|------------|------|---------------|------------|-----------------|---------------|-----------|---------------|------------------|-----------------------|
| **small-8** | 8L × 8H × 512E | 25.7M | 1x RTX 4090 | 32 | 4 | 128 | ~52h | ~52h | 0.294406 | 0.284371 | Smaller baseline |
| **small-16** | 16L × 8H × 512E | 50.9M | 1x RTX 4090 | 32 | 3 | 96 | ~76h | ~76h | 0.272486 | 0.265809 | **Baseline** |
| **small-24** | 24L × 8H × 512E | 76.1M | 2x RTX 4090 | 16 | 8 | 128 | 73.1h | 146.2h | 0.262813 | 0.256293 | +3.6% improvement |
| **small-36** | 36L × 8H × 512E | 113.8M | 2x RTX 4090 | 16 | 8 | 128 | 203.2h | 406.5h | 0.258288 | 0.243144 | +5.2% improvement |
| **medium-12** | 12L × 12H × 768E | 85.8M | 2x RTX 4090 | 32 | 4 | 128 | ~147h | ~294h | 0.265215 | 0.265549 | +2.7% improvement |
| **medium-16** | 16L × 12H × 768E | 114.1M | 2x RTX 4090 | 32 | 4 | 128 | ~147h | ~294h | 0.260812 | 0.255198 | +4.3% improvement |
| **large-16** | 16L × 16H × 1024E | 202.4M | 2x RTX 4090 | 32 | 4 | 128 | 147.8h | 295.7h | 0.257796* | 0.238053 | **+5.4% best overall** |

#### 4.2.4 Training Dynamics and Convergence

**Visual Analysis:**
![Raw Validation Loss Curves](val_loss_raw.png)
![Smooth Validation Loss Curves](val_loss_smooth.png)

**Convergence Characteristics:**
- **Consistent scaling**: Validation loss decreases with parameter count across all architectures
- **No overfitting**: All models show continued learning potential at 600k iterations
- **Depth advantage**: small-36 nearly matches large-16 performance despite 44% fewer parameters
- **Training stability**: fragmented W&B runs for small-8/16 due to technical issues, but final performance validated

**Key Architectural Insight:**  
> Thinner but deeper networks demonstrate superior parameter efficiency for chess reasoning, potentially due to the sequential nature of chess move processing requiring hierarchical feature refinement across layers.

### 4.3 Training Dynamics

**Convergence Patterns:**
[PLACEHOLDER - REQUIRES WANDB DATA DOWNLOAD: Analysis of training curves, convergence speeds, and loss dynamics across different model sizes.]

**Memory and Compute Efficiency:**
- All models trained successfully on 1-2x RTX 4090 setup
- Larger models required more gradient accumulation steps to fit in memory
- Training times scaled approximately linearly with parameter count

Following Karvonen's setup, all models were trained with dropout=0.0. Despite this lack of explicit regularization, no overfitting was observed, suggesting the chess domain and large dataset size (16M games) provide sufficient implicit regularization.

## 5. ⚙️ Technical Implementation

**NanoGPT Modifications:**
- Used standard NanoGPT architecture with minimal modifications
- 32-token vocabulary for chess PGN characters (pieces, squares, notation symbols)
- Block size: 1024 characters, context length: 1023 tokens
- Modified data loading to respect game boundaries (`;` delimiter)


**Training Configuration Management:**
Each model variant has a dedicated configuration file in `config/`:
- `train_stockfish_small_8.py`, `train_stockfish_small_16.py`, etc.
- Hyperparameters version-controlled for full reproducibility
- Command-line overrides supported for quick experiments

**Memory and Compute Optimizations:**
- Mixed precision (FP16) training for memory efficiency
- Gradient accumulation to achieve larger effective batch sizes
- Distributed Data Parallel (DDP) across 1-2x RTX 4090 GPUs (model-dependent)
- Batch sizes adjusted per model to maximize GPU utilization

**Gradient Accumulation Strategy:**

We observed an interesting pattern where **wider models required lower gradient accumulation** compared to **deeper models** for similar effective batch sizes:

| Architecture Type | Example | Per-GPU Batch | Grad Accumulation | Memory Constraint |
|------------------|---------|---------------|-------------------|-------------------|
| **Wide & Shallow** | large-16 (16L×16H×1024E) | 32 | 4 | Parameter memory |
| **Deep & Narrow** | small-36 (36L×8H×512E) | 16 | 8 | Activation memory |

This occurred because:
- **Deep models** hit **activation memory limits** first (storing gradients across many layers)
- **Wide models** hit **parameter memory limits** but have activation headroom
- RTX 4090's 24GB VRAM allowed larger batch sizes for architectures with fewer intermediate activations

This memory efficiency difference may contribute to the observed training dynamics and architectural performance patterns beyond pure parameter scaling.


**Monitoring and Logging:**
- Weights & Biases integration for experiment tracking
- Validation loss computed every 4,000 training steps
- Best checkpoints saved based on validation loss
- Training curves and metrics logged for analysis

> All experiments tracked at [wandb.ai/jd0g-the-university-of-manchester/chess-gpt](https://wandb.ai/jd0g-the-university-of-manchester/chess-gpt/workspace)

## 6. 🚀 Reflection and Next Steps

### 6.1 What This Training Study Accomplished

This systematic training study successfully established a robust foundation for investigating how transformer models learn chess through next-token prediction. Key accomplishments include:

**Experimental Validation:**
- Successfully replicated Karvonen's baseline results, confirming experimental setup.
- Trained 7 model variants spanning 25M to 202.4M parameters 
- Demonstrated consistent scaling trends in validation loss with parameter count
- Established stable training dynamics across all model sizes


**Empirical Observations:**
- Some models appear to approach convergence between 500-600K iterations
- Validation loss improvements correlate with parameter count
- Initial evidence suggests potential depth efficiency in similar parameter regimes
- Training remained stable across all architectural configurations tested

**Scaling Law Implications:** Our results suggest domain-specific scaling may follow different patterns than general language modeling. The vocabulary compression enables exploring deeper/wider architectures within parameter budgets, potentially revealing architectural preferences (depth vs width) that are masked in standard transformer scaling studies where vocabulary dominates parameter allocation.

### 6.2 Questions Raised for Future Investigation

This training work naturally leads to several important research questions for downstream analysis:

**Scaling and Architecture:**
- Do the observed depth vs. width trends hold under rigorous statistical analysis?
- How do these scaling patterns translate to actual chess performance benchmarks?
- What mechanisms drive the apparent depth efficiency, if validated?

**Internal Representations:**
- How do the learned representations evolve across different model sizes?
- At what layers do chess-specific concepts emerge in deeper vs. wider models?
- Do architectural differences lead to qualitatively different internal world models?

**Chess-Specific Performance:**
- How do training loss improvements translate to move quality and game performance?
- Where do these models excel or fail in different phases of chess games?
- What chess concepts are most challenging for different architectural choices?

### 6.3 Connection to Broader Research Program

> This training foundation enables the broader mechanistic interpretability investigation outlined in the thesis research.

The systematic model variants provide the necessary substrate for:
- **Linear probing studies** to investigate internal board state representations
- **Intervention experiments** to establish causal relationships between internal states and outputs  
- **Granular chess diagnostics** to understand how scaling affects chess-specific capabilities
- **Architectural comparisons** in how different models learn and represent chess concepts



---

## References

¹ Nanda, N. (2023). *Actually, Othello-GPT Has A Linear Emergent World Representation*. [Alignment Forum](https://www.alignmentforum.org/posts/nmxzr2zsjNtjaHh7x/actually-othello-gpt-has-a-linear-emergent-world). See also: [Othello-GPT: Reflections on the Research Process](https://www.alignmentforum.org/posts/TAz44Lb9n9yf52pv8/othello-gpt-reflections-on-the-research-process).

² Karvonen, A. (2024). *A Chess-GPT Linear Emergent World Representation*. [Blog post](https://adamkarvonen.github.io/machine_learning/2024/01/03/chess-world-models.html). Code and models: [GitHub](https://github.com/adamkarvonen/chess_llm_interpretability).

³ Karpathy, A. (2022). *nanoGPT*. [GitHub repository](https://github.com/karpathy/nanoGPT). "The simplest, fastest repository for training/finetuning medium-sized GPTs."