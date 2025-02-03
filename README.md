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
- **Dataset**: Replace with your dataset (`nancy9/labels-characters` in example) This is CoDraw dataset we uploaded it on hugging face

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

  # Interpolation Final Code

The VFI is based on RIFE and SAFA. We aim to enhance their practicality for users by incorporating various features and designing new models. Since improving the PSNR index is not consistent with subjective perception, this project is intended for engineers and developers. For general users, we recommend the following software:

- **SVFI (中文)** | **RIFE-App** | **FlowFrames**

Thanks to the SVFI team for supporting model testing on Animation.

## Related Tools

- **VapourSynth-RIFE** | **RIFE-ncnn-vulkan** | **VapourSynth-RIFE-ncnn-Vulkan** | **vs-mlrt** | **Drop frame fixer and FPS converter**

## Frame Interpolation

**2024.08** - We find that **4.24+** is quite suitable for post-processing of some diffusion model-generated videos.

## Trained Models

The content of these links is under the same MIT license as this project. *Lite* models use a similar training framework but have a lower computational cost. Currently, **4.25** is recommended for most scenes.

- **4.26** - 2024.09.21 | [Google Drive](#) | [Baidu](#) | **4.26.heavy** | **4.25.lite** - 2024.10.20
- **4.25** - 2024.09.19 | [Google Drive](#) | [Baidu](#)  
  Improved flow blocks, significantly enhancing anime scene interpolation.
- **4.22** - 2024.08.08 | [Google Drive](#) | [Baidu](#) | **4.22.lite**
- **4.21** - 2024.08.04 | [Google Drive](#) | [Baidu](#)
- **4.20** - 2024.07.24 | [Google Drive](#) | [Baidu](#)
- **4.18** - 2024.07.03 | [Google Drive](#) | [Baidu](#)
- **4.17** - 2024.05.24 | [Google Drive](#) | [Baidu](#)  
  Added gram loss from FILM | **4.17.lite**
- **4.15** - 2024.03.11 | [Google Drive](#) | [Baidu](#) | **4.15.lite**
- **4.14** - 2024.01.08 | [Google Drive](#) | [Baidu](#) | **4.14.lite**
- **v4.9.2** - 2023.11.01 | [Google Drive](#) | [Baidu](#)
- **v4.3** - 2022.08.17 | [Google Drive](#) | [Baidu](#)
- **v3.8** - 2021.06.17 | [Google Drive](#) | [Baidu](#)
- **v3.1** - 2021.05.17 | [Google Drive](#) | [Baidu](#)

[More Older Versions](#)

## Installation

Ensure you have **Python <= 3.11** installed.

```bash
git clone git@github.com:hzwer/Practical-RIFE.git
cd Practical-RIFE
pip3 install -r requirements.txt
```

Download a model from the list above and place `*.py` and `flownet.pkl` in the `train_log/` directory.

## Running the Model

You can use our demo video or your own video for interpolation.

### Basic Usage

```bash
python3 inference_video.py --multi=2 --video=video.mp4
```
This generates `video_2X_xxfps.mp4` with 2X interpolation.

```bash
python3 inference_video.py --multi=4 --video=video.mp4
```
For 4X interpolation.

```bash
python3 inference_video.py --multi=2 --video=video.mp4 --scale=0.5
```
If your video has a high resolution (e.g., 4K), we recommend setting `--scale=0.5` (default is `1.0`).

```bash
python3 inference_video.py --multi=4 --img=input/
```
To read video from PNG images (`input/0.png` ... `input/612.png`). Ensure the PNG filenames are numerical.

## License

This project is licensed under the **MIT License**.



