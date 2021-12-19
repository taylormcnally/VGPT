<p align="center">
   <img src="./resources/voxelgan.png" height="40%" width="40%"/>
   <br>
   <a>
      <img src="https://img.shields.io/badge/python-3.9-blue.svg" alt="Gitter">
   </a>
   <a>
      <img src="https://camo.githubusercontent.com/7ce7d8e78ad8ddab3bea83bb9b98128528bae110/68747470733a2f2f616c65656e34322e6769746875622e696f2f6261646765732f7372632f74656e736f72666c6f772e737667" alt="Gitter">
   </a>
   <a href="https://opensource.org/licenses/MIT">
      <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="Gitter">
   </a>
   <h3 align="center">Tensorflow 2 4D-GAN</h3>
</p>


## Summary

Streamline the creation of high performance GANs for voxels, videos or anything a 4D-GAN would be useful for!

## **Currently in development**


## Requirements

Hardware:

These recommendations are for the default settings but give you a rough idea of what is expected.

Inferencing: 16GB VRAM

Training:  >40GB VRAM


## Data

There are two different types of GANs depending on the input dataset

### VidGAN

*RGB Video Data*

You have a couple of options when it comes to data. The expected data file format is name ordered '.png', however there is logic for parsing images from '.mp4' formats.

1. Find a mp4 or png files of video data
2. Process the mp4 files with worker threads (If not in png format)
3. Run training on the processed dataset

### VoxGAN

*Voxel or pointcloud data*

*To be completed*

## References

StyleGAN3 architecture

```text
@inproceedings{Karras2021,
  author = {Tero Karras and Miika Aittala and Samuli Laine and Erik H\"ark\"onen and Janne Hellsten and Jaakko Lehtinen and Timo Aila},
  title = {Alias-Free Generative Adversarial Networks},
  booktitle = {Proc. NeurIPS},
  year = {2021}
}

```
