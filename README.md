# Tiny ML Model Optimization with PyTorch

## Overview
Model optimization toolkit that is necessary for quantization for 2bit/4bit/8bit weights in QAT(Quantization Aware Training)/PTQ(Post Training Quantization) flows for TI devices with or without NPU. The following package provides wrappers for models running on TI-NPU (HW Accelerator) or CPU. The wrappers for TI-NPU starts from TINPU whereas for CPU starts from GENERIC.

* [Tiny ML TorchModelOpt Documentation](./torchmodelopt/) - Tools and utilities to help the development of embedded Models in [Pytorch](https://pytorch.org) - we call these **model optimization tools**.
* This repository helps you to quantize (with Quantization Aware Training - QAT or Post Training Quantization - PTQ) your model to formats that can run optimally on TI's MCUs

## Installation Instructions

If you want to use the repository as it is, i.e a Python package, then you can simply install this as a pip installable package:

```commandline
pip install git+https://github.com/TexasInstruments/tinyml-tensorlab.git@main#subdirectory=tinyml-modeloptimization/torchmodelopt
```

To setup the repository for development, this python package and the dependencies can be installed by using the setup file.

```commandline
cd tinyml-modeloptimization/torchmodelopt
./setup.sh
```

## Features

The repository provides the following features:
1. **Examples**: The repository comes with examples to get started with modeloptimization
2. **Different Bit-Widths**: Lower precision for representing weights, biases and numbers can be selected to save memory and speed up the inference
3. **PTQ/QAT**: Different quantization methods like PTQ and QAT are supported
4. **ONNX Models**: Quantized models are exported as ONNX structure which can be easily compiled and run on device

Examples for using this repository is present at [Examples](./torchmodelopt/examples/) and for compilation of ONNX Models using TVM Compiler at [Compilation](https://jenkins-sdomc.dal.design.ti.com/job/build-tvm-tinie/lastSuccessfulBuild/artifact/bem/neo-tvm/ti_docs/build/compiling.html)

## Directory Structure

```
tinyml-modeloptimization/
    ├── torchmodelopt/
        ├── examples/
            ├── audio_keyword_spotting/                     # Advanced example on Google Speech Dataset
            ├── fmnist_image_classification/                # Simple example on Fashion MNIST Dataset
            ├── motor_fault_time_series_classification/     # Moderate example on MotorFault Dataset
            ├── README.md    
        ├── requirements/
            ├── requirements.txt
        ├── tinyml_torchmodelopt/
            ├── nas/                                        # Neural Architecture Search
            ├── quantization/                               # Scripts handling PTQ/QAT quantization
                ├── base/                                   # Base Wrapper for Quantized Model
                ├── generic/                                # Constraints for CPU Quantized Model
                    ├── quant_fx.py
                    ├── quant_utils.py
                ├── tinpu/                                  # Constraints for NPU Quantized Model
                    ├── quant_fx.py
                    ├── quant_utils.py
            ├── surgery/                                    # Scripts for replacing PyTorch layers
        ├── setup.py
        ├── setup.sh
    ├── LICENSE
    ├── README.md
```

**Note**: Some files and folders are not represented in this dir structure to avoid cluttering and removing unnecessary information.

## Wrappers Provided

### 1. Base Wrapper for Native PyTorch Quantization
* **TinyMLQuantFxBaseModule**
    - Base class for Generic and TINPU wrappers
    - Model is present in ONNX Format
    - Quantized according to **is_qat** option

### 2. GENERIC Wrappers for CPU Quantization
* **GenericTinyMLQATFxModule** & **GenericTinyMLPTQFxModule**
    - Runs on CPU
    - Model is present in ONNX QDQ Format
    - Quantized according to QAT/PTQ

### 3. TINPU Wrappers for NPU Quantization
* **TINPUTinyMLQATFxModule** & **TINPUTinyMLPTQFxModule**
    - Runs on NPU
    - Model is present in ONNX TINPU Format
    - Quantized according to QAT/PTQ


## Options present in TinyMLQuantFxBaseModule

```python

ti_model = TinyMLQuantFxBaseModule(model, 
                                   qconfig_type=None,
                                   example_inputs=None, 
                                   is_qat=True, 
                                   backend="qnnpack",
                                   total_epochs=0, 
                                   num_batch_norm_update_epochs=None, 
                                   num_observer_update_epochs=False,
                                   prepare_qdq=True,
                                   bias_calibration_factor=0.0, 
                                   verbose=True, 
                                   float_ops=False)

```

#### Argument Descriptions

| Argument                  | Type      | Description |
|---------------------------|-----------|-------------|
| **model**                   | torch.nn.Module       | Model |
| **qconfig_type**     | QConfigMapping/QConfig       | QConfig configurations for model quantization |
| **example_inputs**           | torch.Tensor       | Example input with batch size 1|
| **is_qat**            | bool       | Toggle for PTQ / QAT |
| **backend**           | str       | Backend used to run model |
| **total_epochs**  | int      | Total number of quantized training epochs |
| **num_batch_norm_update_epochs**   | bool/int       | Whether freezing BatchNorm allowed or not, if yes, then provide number of epochs after freezing happens |
| **num_observer_update_epochs**        | bool/int       | Whether freezing observers allowed or not, if yes, then provide number of epochs after freezing happens |
| **prepare_qdq**   | bool       | Extract the pytorch qdq model |
| **bias_calibration_factor**                    | float     | Use bias calibration |
| **verbose**              | bool     | Enable or disable verbose statements |
| **float_ops**          | bool     | Enable float bias for Conv and Linear layers, increases accuracy and inference time |


## Tips & Notes

- **num_batch_norm_update_epochs**
    - None: Freezes the BatchNorm in middle of epoch
    - False: Doesn't freeze the BatchNorm which will overfit the model
    - int (epoch): Best to keep the value from half or 3/4th epoch
- **float_ops**
    - If enabled the addition will have float bias which increases the accuracy
    - This disables the BNORM to happen on TINPU HW

---