# TerraSense

![alt text](https://github.com/mvgianel/TerraSense-AMD-Contest/blob/main/src/imgs/TerraSenseGraphPaper.png)

## TerraSet and ResNet18 Training on TerraSet

### TerraSet

TerraSet6 can be downloaded from: https://drive.google.com/file/d/1CkXGE8A2A4KFMvkAKDVscQWaFebh2V86/view?usp=drive_link .
The dataset consists of 6 categories (cobblestone/brick, dirtground, grass, pavement, sand and stairs) with 100 images each.
In order to have more data, we do data augmentation through code.
We aim to extend the dataset with more categories but also with more images.

### ResNet18 Training on TerraSet

**Disclaimer**: The scripts are originally from the Vitis-AI-Tutorial repo (https://github.com/Xilinx/Vitis-AI-Tutorials/tree/3.5). We have modified them to load, process and augment TerraSet and train the ResNet18 on this dataset.

In order to run this example, please follow the tutorial from the Vitis-AI-Tutorial repo with the following changes:
* The resnet18_terraset folder from this repo needs to be placed in the Vitis-AI-Tutorial/Tutorials/ folder.
* The dataset (TerraSet6) needs to be downloaded, unzipped and placed in Vitis-AI/Vitis-AI-Tutorial/Tutorials/resnet18_terraset/files/target/terraset/. The name of the folder needs to be terraset6.
* After running the appropriate docker as per the original tutorial, make sure to activate the conda environment, then run the following commands:
  - source run_all_2.sh run_clean_dos2unix
  - source run_all_2.sh terraset6_dataset
  - source run_all_2.sh run_terraset6_training_and_quantization
  - source run_all_2.sh compile_resnet18_terraset6

These scrips build the augmented dataset, train ResNet18 (wihout pre-training) and quantize and compile the model for deployment on the DPU. The .xmodel can be found in the build folder.

## ROS2 TerraSense Package

### Installation
#### Simulation Instalation
* Ros-humble needs to be installed on a laptop
* Follow this guide to prepare the Kria https://github.com/amd/Kria-RoboticsAI?tab=readme-ov-file
* Clone repository into your workspace (if you followed the Kria RobotisAI guide, this should be called ros2_ws/src)
* In your workspace (ros_w) run the commands
````
rosdep init
rosdep update
apt install ros-humble-imu-filter-madgwick
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source /opt/ros/humble/setup.bash
source install/setup.bash
````
#### To run on a robot

### Running the code
To run just the inference on the FPGA use the command
````
ros2 run terra_sense terrain_publisher.py
````
To run just the inferrence on the CPU use the command 
````
ros2 run terra_sense terrain_publisher_cpu.py
````
If you have rosbags, you can run them 
````
ros2 bag play <rosbag_folder>
````
To run the inference plus necessary nodes for robot operation use
````
ros2 launch terra_sense terrasense_kria.launch.py launch_terra_sense:=true 
````

  
