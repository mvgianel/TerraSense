# TerraSense - AMD Pervasive AI Developer Contest

![alt text](https://github.com/mvgianel/TerraSense-AMD-Contest/blob/main/src/imgs/TerraSenseGraphPaper.png)

## TerraSet and ResNet18 Training on TerraSet

### TerraSet

TerraSet6 can be downloaded from: https://drive.google.com/file/d/1CkXGE8A2A4KFMvkAKDVscQWaFebh2V86/view?usp=drive_link .
The dataset consists of 6 categories (cobblestone/brick, dirtground, grass, pavement, sand and stairs) with 100 images each.
In order to have more data, we do data augmentation through code.
We aim to extend the dataset with more categories but also with more images.

### ResNet18 Training on TerraSet

**Disclaimer**: The scripts are originally from the Vitis-AI-Tutorial repo (https://github.com/Xilinx/Vitis-AI-Tutorials/tree/3.5). We have modified them to load, process and augment TerraSet and train the ResNet18 on this dataset.
We have created this separate repo for the AMD Pervasive AI Developer Contest.

In order to run this example, please follow the tutorial from the Vitis-AI-Tutorial repo with the following changes:
* The resnet18_terraset folder from this repo needs to be placed in the Vitis-AI-Tutorial/Tutorials/ folder.
* The dataset (TerraSet6) needs to be downloaded, unzipped and placed in Vitis-AI/Vitis-AI-Tutorial/Tutorials/resnet18_terraset/files/target/terraset/. The name of the folder needs to be terraset6.
* After running the appropriate docker as per the original tutorial, make sure to activate the conda environment, then run the following commands:
  - source run_all_2.sh run_clean_dos2unix
  - source run_all_2.sh terraset6_dataset
  - source run_all_2.sh run_terraset6_training_and_quantization
  - source run_all_2.sh compile_resnet18_terraset6

These scrips build the augmented dataset, train ResNet18 (wihout pre-training) and quantize and compile the model for deployment on the DPU. The .xmodel can be found in the build folder.


  
