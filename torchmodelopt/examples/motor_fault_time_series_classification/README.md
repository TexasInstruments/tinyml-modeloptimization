# Motor Fault Time Series Classification

Motor Fault Dataset is a subset of motor fault dataset prepared by TIs Internal Team. The dataset consists of vibrations received from running motors. The motors can be classified as Normal, Localized Fault, Erosion Fault, Flaking Fault. The dataset has 4800030 samples. Each sample has 4 variables and 1 target.

This example will use a Deep Learning model to train and classify the type of motor fault.

## Walkthrough of this Example
1. Create train and test dataloader from csv
2. Configure the training and quantization params
4. Wrap the trained model around TinyMLFxModule
5. Train and test this ti_model for `QAT`/`PTQ`
6. Export the quantized model

## Let's understand each step

### Prepare Dataloader
### Configure Quantization
### Using Quantization
### Train and Test Quantization on CNN Model
### Exporting the quantized model