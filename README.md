# Preference Optimization for Language Diffusion Models

Traditional preference optimization methods like DPO face computational challenges when applied to diffusion language models due to the intractable nature of computing exact log probabilities. This implementation provides a practical solution using Monte Carlo estimation with Evidence Lower Bound (ELBO) approximation.

## Experimental Setup

### Model Configuration

- **Base Model**: LLaDA-8B-Instruct
- **Quantization**: 8-bit with BitsAndBytesConfig
- **LoRA Config**:
  - Rank (r): 8
  - Alpha: 32
  - Dropout: 0.1
  - Target modules: ["q_proj", "k_proj", "v_proj", "attn_out"]

### Training Configuration

- **Dataset Size**: 1,024 samples (GPU constrained)
- **Batch Size**: 1 (with gradient accumulation of 64)
- **Effective Batch Size**: 64
- **Epochs**: 1
- **Learning Rate**: 5e-6
- **Scheduler**: Linear warmup + Cosine annealing
- **Monte Carlo Timesteps**: 8

### Key Implementation Choices

1. **Constant Timesteps**: Timesteps are kept constant for each loss calculation to reduce variance in the Monte Carlo estimation.

2. **Gradient Checkpointing**: Used to reduce memory consumption during forward passes.

3. **Mixed Precision**: BF16 training for improved memory efficiency and speed.

4. **Reference Model Freezing**: Reference model parameters are frozen to prevent drift during training.

## Limitations and Future Work

### Current Limitations

- Small dataset size due to GPU constraints
- Limited to 1 epoch of training
- Monte Carlo approximation introduces estimation noise

### Future Improvements

- Scale to larger datasets with better hardware
- Implement variance reduction techniques from VRPO
- Experiment with different timestep sampling strategies
- Evaluate on larger test sets

## References

1. **Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C.** (2023).
   Direct preference optimization: Your language model is secretly a reward model. arXiv preprint arXiv:2305.18290

2. **Nie, S., Zhu, F., You, Z., Zhang, X., Ou, J., Hu, J., Zhou, J., Lin, Y., Wen, J.-R., & Li, C.** (2025).
   Large language diffusion models. arXiv preprint arXiv:2502.09992

3. **Zhu, F., Wang, R., Nie, S., Zhang, X., & Wei, C.** (2025).
   LLaDA 1.5: Variance-reduced preference optimization for large language diffusion models. arXiv preprint arXiv:2505.19223

_Note: The official code for VRPO (Variance-Reduced Preference Optimization) had not been released at the time of this implementation._

## Acknowledgments

- The LLaDA team for providing the base diffusion language model
- The DPO authors for the original preference optimization framework
- The transformers and accelerate libraries for efficient model training infrastructure
