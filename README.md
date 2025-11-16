# Simplified Implementation of Conditional Diffusion Model (MNIST Hand-write Number Generation)

---
🌐 **Select Language / 选择语言:**
- [English](README.md)
- [简体中文](README.zh_cn.md)
---

This repository implements a Conditional Diffusion Model, which can be considered a minimal, educational version of models like Stable Diffusion. The model is trained on the classic MNIST handwritten digit dataset. It is a product of my personal learning, and I hope releasing it can help more beginners deeply understand the core principles of diffusion models. 🙂

## Model Features

- **Conditional Generation**: Supports generating corresponding handwritten digit images based on digit labels (0-9)

- **Simplified Architecture**: A Conditional Diffusion Model based on a U-Net, with clear and easy-to-understand code structure

- **Complete Implementation**: Includes the entire pipeline: forward diffusion, reverse denoising, training, and generation

## Core Components

1.  Scheduler: Controls the noise addition process
2.  TimestepEmbedding: Converts discrete timesteps into continuous vectors
3.  Conditional U-Net: Feature extraction network that supports conditional input
4.  Training Pipeline: Complete model training and loss calculation

## Usage

```bash
# Train the model
python main.py

# Generate digits (Interactive)
# Input a digit label (0-9), and the model will generate the corresponding handwritten digit
```

`diffusion.pkl` contains a pre-trained model. You can run `main.py` to see the results directly without training.

## Important Notes

⚠️ **GPU Support**: If you need GPU acceleration, please do not install PyTorch via `requirements.txt`. Instead, get the appropriate installation command for your CUDA version from [https://pytorch.org](https://pytorch.org).

## Contribution

If you find this repository helpful, please kindly give us a free ⭐.(\*^ω^\*)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
