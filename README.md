# Image Generation and Inpainting Tool
This repository provides a tool for generating images that align with a given text prompt while incorporating an unaltered object image into the scene. The tool ensures the generated images look natural and can also create videos from these images using Stable Diffusion models. The solution leverages publicly available pre-trained checkpoints and various libraries to achieve this goal.

[click here](https://docs.google.com/document/d/14c85QpdgfYlPWbJ1camb6EvhMCtgehugWUtPL2xLTv4/edit?usp=sharing)
 for detailed documentation.
 
## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)

## Prerequisites

- Python 3.7 or higher
- CUDA-enabled GPU (recommended for faster processing)

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/image-generation-inpainting-tool.git
   cd image-generation-inpainting-tool

2. ```
    pip install -r requirements.txt

## Usage
    python run.py --image <path_to_input_image> --text-prompt "<text_prompt>" --output <path_to_save_generated_image> [--debug] [--video_output_path <path_to_save_video>]

### Arguments

- `--image`: Path to the input image.
- `--text-prompt`: Text prompt for image generation.
- `--output`: Path to save the generated image.
- `--debug` (optional): Enable debug mode to save intermediate images.
- `--video_output_path` (optional): Path to save the generated video.



