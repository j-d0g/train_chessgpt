# ChessGPT Model Documentation

This directory contains documentation and tools for the ChessGPT model series. The comprehensive model card for all variants has been published to Hugging Face Hub.

## 🏆 Official Model Card

**📍 Primary Location:** [Hugging Face Hub - jd0g/chess-gpt](https://huggingface.co/jd0g/chess-gpt)

The unified model card covers all 7 trained model variants with verified specifications and performance metrics.

## 📊 Actual Model Variants (Verified)

### 🤖 Stockfish-Trained Models (7 variants)
| Model | Parameters | Layers | Heads | Embedding | Val Loss | Architecture Focus |
|-------|------------|--------|-------|-----------|----------|-------------------|
| **small-8** | 25.7M | 8 | 8 | 512 | 0.294406 | Minimal baseline |
| **small-16** | 50.9M | 16 | 8 | 512 | 0.272486 | Depth scaling |
| **small-24** | 76.1M | 24 | 8 | 512 | 0.262813 | Deep narrow |
| **small-36** | 113.8M | 36 | 8 | 512 | 0.258288 | Very deep narrow |
| **medium-12** | 85.8M | 12 | 12 | 768 | 0.265215 | Balanced medium |
| **medium-16** | 114.1M | 16 | 12 | 768 | 0.260812 | Deeper medium |
| **large-16** | 202.4M | 16 | 16 | 1024 | **0.257796*** | **Best overall** |

***Note:** The large-16 model actually achieved a best validation loss of **0.25621** during training, but due to a checkpointing bug, the final saved checkpoint corresponded to iteration 600k rather than the best validation loss checkpoint.

### 👥 Human vs. Synthetic Data Analysis
Adam Karvonen's work included training on human games (Lichess dataset) in addition to Stockfish self-play data. Key findings from this comparison:

- **Human-trained models**: Achieved lower validation loss but worse chess performance
- **Stockfish-trained models**: Higher validation loss but better chess performance  
- **Interpretation**: Stockfish data contains more complex patterns that are harder to model but lead to superior play. Human games may be more predictable (similar mistake patterns) but contain modeling imperfections absent in engine play.
- **Architectural Alignment**: This suggests GPT architectures may be more aligned with human thinking patterns (attention, pattern recognition, heuristics) compared to Stockfish's brute-force tree search approach.

Our study focused on Stockfish-trained models to investigate the upper limits of chess capability in transformer architectures.

## 🔬 Key Research Contributions

### 📈 1. Parameter Scaling Laws (Verified)
Our systematic scaling study demonstrates clear benefits:
- **25.7M → 202.4M parameters:** Consistent validation loss improvements across the range
- **No overfitting observed:** All models continued learning at 600k iterations  
- **Smooth scaling curve:** Consistent improvements across parameter range

### 🏗️ 2. Architecture Insights (Depth vs. Width)
Comprehensive exploration of transformer architectures for chess:
- **Layer Depth Tested**: 8, 12, 16, 24, 36 layers
- **Attention Heads**: 8, 12, 16 heads across different scales  
- **Embedding Dimensions**: 512, 768, 1024 dimensions
- **Key Finding**: **Depth > Width** for chess reasoning (deeper narrow models outperform wider shallow models)

### ⚡ 3. Training Convergence Patterns
Observed convergence behavior across model variants:

| Convergence Order | Model | Convergence Behavior |
|-------------------|-------|---------------------|
| **Fastest** | large-16 | Converges ~500k, flattens completely |
| | medium-16 | Approaches halt by 600k |
| | medium-12 | Approaches halt by 600k |
| | small-8 | Slowly starts to converge |
| | small-24 | Still converging but slowed rate |
| | small-36 | Still converging but slowed rate |
| **Slowest** | small-16 | Continued linear descent through 600k |

**Key Finding**: Larger models converge faster but smaller models show continued potential for improvement with extended training.

### 📚 4. Dataset Comparison
Engine vs. human gameplay patterns:
- **Stockfish Dataset**: High-quality engine games (primary dataset)
- **Lichess Dataset**: Human games for behavioral comparison
- **Research Value**: Understanding human vs. optimal play patterns

## Technical Specifications

### Common Architecture Features
- **Base Model**: GPT-2 transformer architecture
- **Framework**: NanoGPT (PyTorch implementation)
- **Tokenization**: Chess-specific vocabulary (32 tokens)
- **Input Format**: PGN games in 1024-character blocks
- **Delimiter**: ";" token to mark game beginnings
- **Training**: Mixed precision (bfloat16/float16)

### Training Configuration
- **Optimizer**: AdamW with β₂=0.95
- **Learning Rate**: 3e-4 with cosine decay to 3e-5
- **Max Iterations**: 600,000 across all models
- **Data Split**: 99% training, 1% validation
- **Hardware**: RTX 4090 GPUs with distributed training

## Dataset Details

### Stockfish Dataset (`stockfish_dataset_blocks.zip`)
```
Size: 4.5GB
Games: Engine-generated
White: Stockfish ELO 3200 (fixed)
Black: Stockfish ELO 1300-3200 (variable)
Purpose: Optimal play patterns and tactical precision
```

### Lichess Dataset (`lichess_6gb_blocks.zip`)
```
Size: 6GB
Games: 16 million human games
Source: Lichess platform (2016-2017)
ELO Range: All skill levels (no filtering)
Purpose: Human gameplay patterns and decision-making
```

## 🎯 Model Performance & Usage

### 🏆 Recommended Models
| Model | Parameters | Val Loss | Best Use Case | Computational Requirements |
|-------|------------|----------|---------------|---------------------------|
| **🥇 large-16** | 202.4M | **0.257796*** | **Best overall performance** | GPU recommended |
| **🥈 small-36** | 113.8M | 0.258288 | Best depth/parameter efficiency | GPU recommended |
| **🥉 medium-16** | 114.1M | 0.260812 | Balanced performance | GPU recommended |
| **📊 small-8** | 25.7M | 0.294406 | Fast inference, baseline | CPU/GPU |

***Note:** large-16 achieved 0.25621 validation loss during training but checkpoint saved final iteration instead of best performance.

## 🚀 Usage Guidelines

### 🎯 Model Selection Strategy
- **🔬 Research/Baseline**: Use **small-8** for quick experiments
- **⚖️ Best Efficiency**: Use **small-36** for optimal depth/parameter ratio  
- **🏆 Maximum Performance**: Use **large-16** for best chess reasoning
- **👥 Human Behavior Studies**: Use **lichess-8** for human-like patterns

### 💻 Computational Requirements
- **small-8, lichess-8**: CPU inference possible, GPU recommended
- **small-16, small-24**: GPU recommended for practical use
- **small-36, medium-12/16**: GPU required for reasonable performance  
- **large-16**: High-end GPU recommended (16GB+ VRAM)

### Input Format
All models expect properly tokenized chess games:
```
";1.e4 e5 2.Nf3 Nc6 3.Bb5 a6..."
```
- Must start with ";" delimiter
- Standard PGN algebraic notation
- 1024-character blocks for training data

## Evaluation Metrics

Models should be evaluated on:
1. **Move Legality**: Percentage of legal moves generated
2. **Game Quality**: Coherence of multi-move sequences
3. **Tactical Accuracy**: Recognition of tactical patterns
4. **Strategic Understanding**: Long-term planning capabilities
5. **Opening Knowledge**: Familiarity with chess theory
6. **Endgame Technique**: Performance in simplified positions

## Research Applications

### Scaling Studies
- Parameter scaling effects on chess reasoning
- Optimal architecture configurations
- Training efficiency and convergence patterns
- Scaling law validation in domain-specific models

### Cognitive Science
- Human vs. engine decision-making patterns
- Chess expertise and pattern recognition
- Strategic reasoning in artificial systems
- Comparative analysis of learning paradigms

### AI Development
- Domain-specific language model training
- Emergent reasoning capabilities
- Transfer learning from chess to other domains
- Interpretability of strategic reasoning

## 📄 Documentation & Tools

### 🔧 Available Tools in This Directory
- **`inspect_models.py`**: Script to extract model architecture from checkpoint files
- **`config.json`**: Hugging Face configuration for download tracking
- **`download_tracking_setup.md`**: Guide for setting up download metrics

### 🌐 External Resources
- **🤗 Hugging Face Hub**: [jd0g/chess-gpt](https://huggingface.co/jd0g/chess-gpt) - Complete model card and downloads
- **📈 Training Logs**: [W&B Workspace](https://wandb.ai/jd0g-the-university-of-manchester/chess-gpt/workspace)
- **📚 Dataset**: [adamkarvonen/chess_games](https://huggingface.co/datasets/adamkarvonen/chess_games)

## 📖 Citation

If you use these models in your research, please cite:

```bibtex
@misc{chessgpt-scaling-2024,
  title={ChessGPT: Scaling Transformer Models for Chess},
  author={Your Name},
  year={2024},
  howpublished={Hugging Face Model Hub},
  url={https://huggingface.co/jd0g/chess-gpt}
}

@dataset{chess_games_dataset,
  title={Chess Games Dataset},
  author={Adam Karvonen},
  year={2024},
  url={https://huggingface.co/datasets/adamkarvonen/chess_games}
}
```

## References

- **NanoGPT**: [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- **Chess Dataset**: [@adamkarvonen/chess_games](https://huggingface.co/datasets/adamkarvonen/chess_games)
- **GPT-2 Paper**: [Radford et al., 2019](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- **Scaling Laws**: [Kaplan et al., 2020](https://arxiv.org/abs/2001.08361)

## License

[Specify your license here - e.g., MIT, Apache 2.0, etc.]

---

## ✅ Status Update

**Current State**: The comprehensive model card has been published to **[Hugging Face Hub](https://huggingface.co/jd0g/chess-gpt)** with verified model specifications and performance metrics.