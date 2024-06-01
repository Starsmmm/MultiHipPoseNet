# MultiHipPoseNet

This is a pytorch implementation of MultiHipPoseNet, a multitasking model for structure segmentation and keypoint detection

![image](https://github.com/Starsmmm/MultiHipPoseNet/nets/schematic%20illustration.png)

The schematic illustration of the MuiltHipPoseNet algorithm. The proposed framework comprises a series of interconnected layers, each with a specific purpose. Conv, represents the convolutional layer. The normalization technique employed here is group normalization, while the activation function is Relu. MaxPooling, is responsible for maximum pooling. Up-sampling, is tasked with upsampling feature maps to recover dimensions. ME-GCT, utilizes a hybrid gated attention unit consisting of multiple experts with adjustable parameters for the number of experts. BINK1 and BINK2 represent different kinds of residual modules.

We collecte hip US images of 781 infants aged 0-6 months. Each patient contains one or two images of the left and right legs, for 1355 
images. The images include 568 infants with type I hips without dislocation and 213 infants with type II hips.

## Requirements

* PyTorch
* scikit-learn
* numpy
* labelme == 3.16.5

## Usage

We provide several python scripts that contain functions for data processing, dividing, training, prediction, and evaluation. These scripts must be run in the following order:

1. json_to_dataset.py - Converts json files labeled with labelme into masked png files.
2. voc_annotation.py - Hierarchical cross-validation divides the data into training, validation, and test sets.
3. train.py - training file, click on it to train the model, optionally with or without ME-GCT.
4. predict.py - test file, perform inference evaluation on the training model (5-fold hierarchical validation), optionally with or without visualization of mask and coordinate prediction results.
5. eval.py - A set of evaluation metrics to calculate the results of the model evaluation.

## Acknowledgements

This project utilizes parts of [Bubbliiiing](https://github.com/Bubbliiiing)'s [unet-pytorch](https://github.com/bubbliiiing/unet-pytorch) and includes modifications and improvements made by us. We thank them for their excellent work.

## Issues

Don't hesitate to contact for any feedback or create issues/pull requests.
