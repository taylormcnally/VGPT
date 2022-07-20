<p align="center">
   <br>
   <a>
      <img src="https://img.shields.io/badge/python-3.9-blue.svg" alt="Python">
   </a>
   <a>
      <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="Pytorch">
   </a>
   <a href="https://opensource.org/licenses/MIT">
      <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT">
   </a>
   <h3 align="center">Pytorch MViT</h3>
</p>

## **Currently in development**


## Summary

Streamline the creation of transformer models for voxels, videos or any higher dimensional data.

## Requirements

Hardware:

These recommendations are for the default settings but give you a rough idea of what is expected.

Inferencing: 16GB VRAM

Training:  >40GB VRAM


## Data

### Processing data

*RGB Video Data*

You have a couple of options when it comes to data. The expected data file format is name ordered '.png', however there is logic for parsing images from '.mp4' formats.

1. Find a mp4 or png files of video data
2. Process the mp4 files with ffmpeg (*preferred fast method is using the nvidia compiled binaries to utilize GPU speedup)

3. use tfds to create the new dataset file

> tfds new my_dataset



### Configuring the model


### Pretraining

### Finetuning





