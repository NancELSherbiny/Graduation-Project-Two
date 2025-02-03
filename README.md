# Graduation-Project-Two
This repository includes Advanced text-to-image synthesis using Stable Diffusion , video frame interpolation using RIFE model and a website to integrate our work.  
# README

## Overview
This project fine-tunes the U-Net component of Stable Diffusion using a custom dataset. The model is trained on captions and corresponding images, leveraging gradient checkpointing, mixed precision training, and early stopping for efficiency.

## Requirements
Ensure you have the following installed:
- Python 3.9
- PyTorch
- Hugging Face `transformers`, `datasets`, and `diffusers`
- `torchvision`, `numpy`, `tqdm`, `torch.cuda.amp`
- `os`, `gc`

You can install dependencies using:
```bash
pip install torch torchvision transformers datasets diffusers numpy tqdm
```

## Model and Dataset
- **Model**: `CompVis/stable-diffusion-v1-4`
- **Dataset**: Replace with your dataset (`nancy9/labels-characters` in example)

## Preprocessing
- Tokenizes captions using CLIPTokenizer.
- Applies image transformations (resize, normalize, convert to tensor).
- Maps the preprocessing function to the dataset.
- Splits training data into a subset.

## Training Pipeline
1. **Load Pretrained Components**
   - CLIP text encoder, VAE, U-Net, and scheduler.
2. **Set Device**
   - Uses GPU if available.
3. **Freeze Layers**
   - Trains only output layers of U-Net.
4. **Gradient Accumulation**
   - Uses `GradScaler` for mixed precision training.
5. **Training Process**
   - Encodes images into latent space.
   - Adds noise to latents.
   - Uses text encoder to generate embeddings.
   - Predicts noise residual and computes loss.
   - Saves weights every 5 epochs.
6. **Early Stopping**
   - Stops if validation loss doesn’t improve for 3 consecutive epochs.

## Model Inference
- Loads the trained model from saved weights.
- Uses `StableDiffusionPipeline` to generate an image from a given text prompt.
- Saves the output image as `generated_image.png`.

## Running the Script
To start training:
```bash
python train.py
```
To generate an image:
```bash
python generate.py
```

## Notes
- Adjust batch size and learning rate based on hardware.
- Ensure `lora_weights/` directory exists to store trained weights.
- Fine-tuned weights are saved as `unet_weights_epoch_X.pth`.

