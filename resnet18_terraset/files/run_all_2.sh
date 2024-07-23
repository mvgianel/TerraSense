#!/bin/bash

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:   11 Aug 2023

echo " "
echo "==========================================================================="
echo "WARNING: "
echo "  'run_all.sh' MUST ALWAYS BE LAUNCHED BELOW THE 'files' FOLDER LEVEL "
echo "  (SAME LEVEL OF 'modelzoo' AND 'target' FOLDER)                       "
echo "  AS IT APPLIES RELATIVE PATH AND NOT ABSOLUTE PATHS                  "
echo "==========================================================================="
echo " "

# ===========================================================================
# patch
# ===========================================================================
run_patch(){
tar -xvf patch.tar.gz
cd patch
source ./run_patch.sh
cd ..
}


# ===========================================================================
# remove redundant information from host logfile
# ===========================================================================
LOG_FILENAME=$2
prepare_logfile(){
  #cat  logfile_resnet18_terraset6.txt logfile3_resnet18_terraset6.txt > logfile0_resnet18_terraset6.txt
  #mv logfile0_resnet18_terraset6.txt logfile_resnet18_terraset6.txt
  dos2unix -f ${LOG_FILENAME} #logfile_run_all_7apr2023.txt
  cat ${LOG_FILENAME}  | grep -v "loss: " | tee prova1.txt
  cat prova1.txt | grep -v "100%|" | tee prova2.txt
  cat prova2.txt | grep -v "ETA: " | tee prova3.txt
  cat ./doc/header.txt prova3.txt > logfile_host.txt
  rm -f prova*.txt
}


# ===========================================================================
# analyze DPU graphs for TARGET ZCU102 Terraset6
# ===========================================================================
analyze_graphs(){
echo "----------------------------------------------------------------------------------"
echo "ANALYZING GRAPHS FOR ZCU102"
echo "----------------------------------------------------------------------------------"
source ./scripts/analyze_subgraphs.sh zcu102 q_train1_resnet18_terraset6_final.h5
}


# ===========================================================================
# STEP1: clean and dos2unix
# ===========================================================================
run_clean_dos2unix(){
echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO STEP1]  CLEANING ALL FOLDERS"
echo "----------------------------------------------------------------------------------"
echo " "
source ./scripts/clean_all.sh
}

# ===========================================================================
# STEP2: Prepare Terraset6 Dataset
# ===========================================================================
terraset6_dataset(){
#rm -rf build/dataset
echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO STEP2]  CREATING Terraset6 DATASET OF IMAGES"
echo "----------------------------------------------------------------------------------"
echo " "
# organize Terraset6  data
python code/terraset6_generate_images.py   | tee build/log/terraset6_generate_images.log
}

# ===========================================================================
# STEP3: Train ResNet18 CNNs on Terraset6
# ===========================================================================
run_terraset6_training(){
# floating point model training
echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO STEP3A] Terraset6 TRAINING (way 1)"
echo "----------------------------------------------------------------------------------"
echo " "
python ./code/train1_resnet18_terraset6.py --epochs 50 | tee ./build/log/train1_resnet18_terraset6.log
mv ./build/float/train1_best_chkpt.h5 ./build/float/train1_resnet18_terraset6_best.h5
mv ./build/float/train1_final.h5      ./build/float/train1_resnet18_terraset6_final.h5

echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO STEP3B] Terraset6 TRAINING (way 2)"
echo "----------------------------------------------------------------------------------"
echo " "
python ./code/train2_resnet18_terraset6.py --epochs 50 | tee ./build/log/train2_resnet18_terraset6.log

}

# ===========================================================================
# STEP4: Vitis AI Quantization of ResNet18 on Terraset6
# ===========================================================================
quantize_resnet18_terraset6(){
echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO STEP4A] QUANTIZE Terraset6 TRAINED CNN1 MODELS"
echo "----------------------------------------------------------------------------------"
echo " "
echo "[DB INFO STEP4A-1] MODEL INSPECTION"
echo " "
python  ./code/inspect_resnet18_terraset6.py --float_file ./build/float/train1_resnet18_terraset6_final.h5
mv build/log/inspect_results.txt build/log/inspect_results_train1_resnet18_terraset6_final.txt
mv build/log/model.svg build/log/model_train1_resnet18_terraset6_final.svg
echo " "
echo "[DB INFO STEP4A-2] EFFECTIVE QUANTIZATION OF FINAL-CNN1 MODEL"
echo " "
python  ./code/vai_q_resnet18_terraset6.py   --float_file ./build/float/train1_resnet18_terraset6_final.h5 --quant_file ./build/quantized/q_train1_resnet18_terraset6_final.h5
echo " "
echo "[DB INFO STEP4A-3] EFFECTIVE QUANTIZATION OF BEST-CNN1 MODEL"
echo " "
python  ./code/vai_q_resnet18_terraset6.py   --float_file ./build/float/train1_resnet18_terraset6_best.h5  --quant_file ./build/quantized/q_train1_resnet18_terraset6_best.h5
echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO STEP4B] QUANTIZE Terraset6 TRAINED CNN2 MODEL"
echo "----------------------------------------------------------------------------------"
echo " "
echo "[DB INFO STEP4B-1] MODEL INSPECTION"
echo " "
python  ./code/inspect_resnet18_terraset6.py --float_file ./build/float/train2_resnet18_terraset6_float.h5
mv build/log/inspect_results.txt build/log/inspect_results_train2_resnet18_terraset6_float.txt
mv build/log/model.svg           build/log/model_train2_resnet18_terraset6_float.svg
echo " "
echo "[DB INFO STEP4B-2] EFFECTIVE QUANTIZATION"
echo " "
python  ./code/vai_q_resnet18_terraset6.py   --float_file ./build/float/train2_resnet18_terraset6_float.h5 --quant_file ./build/quantized/q_train2_resnet18_terraset6.h5

}

# ===========================================================================
# STEP5: Vitis AI Compile ResNet18 terraset6 for Target Board
# ===========================================================================
compile_resnet18_terraset6(){
#train1
echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO STEP5A] COMPILE terraset6 QUANTIZED CNN1 MODEL"
echo "----------------------------------------------------------------------------------"
echo " "
source ./scripts/run_compile.sh zcu102  q_train1_resnet18_terraset6_final.h5
source ./scripts/run_compile.sh vck190  q_train1_resnet18_terraset6_final.h5
source ./scripts/run_compile.sh vek280  q_train1_resnet18_terraset6_final.h5
source ./scripts/run_compile.sh vck5000 q_train1_resnet18_terraset6_final.h5
source ./scripts/run_compile.sh v70     q_train1_resnet18_terraset6_final.h5

mv   ./build/compiled_zcu102/zcu102_q_train1_resnet18_terraset6_final.h5.xmodel  ./target/terraset6/zcu102_train1_resnet18_terraset6.xmodel
mv   ./build/compiled_vck190/vck190_q_train1_resnet18_terraset6_final.h5.xmodel  ./target/terraset6/vck190_train1_resnet18_terraset6.xmodel
mv   ./build/compiled_vek280/vek280_q_train1_resnet18_terraset6_final.h5.xmodel  ./target/terraset6/vek280_train1_resnet18_terraset6.xmodel
mv   ./build/compiled_v70/v70_q_train1_resnet18_terraset6_final.h5.xmodel        ./target/terraset6/v70_train1_resnet18_terraset6.xmodel
mv ./build/compiled_vck5000/vck5000_q_train1_resnet18_terraset6_final.h5.xmodel  ./target/terraset6/vck5000_train1_resnet18_terraset6.xmodel

#train2
echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO STEP5B] COMPILE terraset6 QUANTIZED CNN2 MODEL"
echo "----------------------------------------------------------------------------------"
echo " "
source ./scripts/run_compile.sh zcu102  q_train2_resnet18_terraset6.h5
source ./scripts/run_compile.sh vck190  q_train2_resnet18_terraset6.h5
source ./scripts/run_compile.sh vek280  q_train2_resnet18_terraset6.h5
source ./scripts/run_compile.sh vck5000 q_train2_resnet18_terraset6.h5
source ./scripts/run_compile.sh v70     q_train2_resnet18_terraset6.h5

mv   ./build/compiled_zcu102/zcu102_q_train2_resnet18_terraset6.h5.xmodel  ./target/terraset6/zcu102_train2_resnet18_terraset6.xmodel
mv   ./build/compiled_vck190/vck190_q_train2_resnet18_terraset6.h5.xmodel  ./target/terraset6/vck190_train2_resnet18_terraset6.xmodel
mv   ./build/compiled_vek280/vek280_q_train2_resnet18_terraset6.h5.xmodel  ./target/terraset6/vek280_train2_resnet18_terraset6.xmodel
mv ./build/compiled_vck5000/vck5000_q_train2_resnet18_terraset6.h5.xmodel  ./target/terraset6/vck5000_train2_resnet18_terraset6.xmodel
mv         ./build/compiled_v70/v70_q_train2_resnet18_terraset6.h5.xmodel  ./target/terraset6/v70_train2_resnet18_terraset6.xmodel

}

# ===========================================================================
# STEP6: prepare archive for TARGET ZCU102 runtime application for terraset6
# ===========================================================================
prepare_terraset6_archives() {
echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO STEP6] PREPARING terraset6 ARCHIVE FOR TARGET BOARDS"
echo "----------------------------------------------------------------------------------"
echo " "
cp -r target       ./build
cd ./build/dataset/terraset6
tar -cvf test.tar ./test > /dev/null
cp test.tar       ../../../build/target/terraset6/
rm test.tar
cd ../../../
rm -rf ./build/target/imagenet #unuseful at the moment
# zcu102
cp -r ./build/target/  ./build/target_zcu102  > /dev/null
rm -f ./build/target_zcu102/terraset6/vck*_terraset6.xmodel
rm -f ./build/target_zcu102/terraset6/vek*_terraset6.xmodel
rm -f ./build/target_zcu102/terraset6/v70*_terraset6.xmodel
# vck190
cp -r ./build/target/  ./build/target_vck190  > /dev/null
rm -f ./build/target_vck190/terraset6/zcu1*_terraset6.xmodel
rm -f ./build/target_vck190/terraset6/vek2*_terraset6.xmodel
rm -f ./build/target_vck190/terraset6/vck5*_terraset6.xmodel
rm -f ./build/target_vck190/terraset6/v70*_terraset6.xmodel
# vek280
cp -r ./build/target   ./build/target_vek280  > /dev/null
rm -f ./build/target_vek280/terraset6/zcu*_terraset6.xmodel
rm -f ./build/target_vek280/terraset6/vck*_terraset6.xmodel
rm -f ./build/target_vek280/terraset6/v70*_terraset6.xmodel
# vck5000
cp -r ./build/target/  ./build/target_vck5000  > /dev/null
rm -f ./build/target_vck5000/terraset6/zcu1*_terraset6.xmodel
rm -f ./build/target_vck5000/terraset6/vek2*_terraset6.xmodel
rm -f ./build/target_vck5000/terraset6/vck1*_terraset6.xmodel
rm -f ./build/target_vck5000/terraset6/v70*_terraset6.xmodel
# v70
cp -r ./build/target/  ./build/target_v70  > /dev/null
rm -f ./build/target_v70/terraset6/zcu1*_terraset6.xmodel
rm -f ./build/target_v70/terraset6/vek2*_terraset6.xmodel
rm -f ./build/target_v70/terraset6/vck*_terraset6.xmodel
}




# =================================================================================================
# STEP7 (1): prepare imagenet test images: you must have downloaded ILSVRC2012_img_val.tar already
# =================================================================================================
ARCHIVE=./files/modelzoo/ImageNet/ILSVRC2012_img_val.tar
prepare_imagenet_test_images(){

if [ -f "$ARCHIVE" ]; then
  echo "ERROR! $ARCHIVE does exist: you have to download it"
else
  cd ./modelzoo/ImageNet/
  mkdir -p val_dataset
  # expand the archive
  echo "expanding ILSVRC2012_img_val.tar archive"
  tar -xvf ILSVRC2012_img_val.tar -C ./val_dataset > /dev/null
  ls -l ./val_dataset | wc
  python3 imagenet_val_dataset.py
  cd ../..
  # copy the archive to the ``target/imagenet`` folder
  cp ./modelzoo/ImageNet/val_dataset.zip ./target/imagenet
  cd ./target/imagenet/
  unzip -o -q val_dataset.zip #unzip forcing overwrite in quiet mode
  cd ../../
fi
}

# ===========================================================================
# STEP7 (2): Vitis AI Quantization of ResNet50 on ImageNet
# ===========================================================================
quantize_resnet50_imagenet(){
echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO STEP7] IMAGENET RESNET50: EVALUATE & QUANTIZE"
echo "YOU SHOULD HAVE ALREADY DOWNLOADED tf2_resnet50_3.5.zip ARCHIVE"
echo "----------------------------------------------------------------------------------"
echo " "
DIRECTORY1=./files/modelzoo/tf2_resnet50_3.5

if [ -d "$DIRECTORY1" ]; then
    echo "ERROR! $DIRECTORY1 does exist: cannot evaluate ResNet50 CNN!"
else
    python  ./code/eval_resnet50.py
fi
}

# ===========================================================================
# STEP8: Vitis AI Quantization of ResNet18 on ImageNet
# ===========================================================================
quantize_resnet18_imagenet(){
echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO STEP8] IMAGENET RESNET18: EVALUATE & QUANTIZE"
echo "----------------------------------------------------------------------------------"
echo " "
python  ./code/eval_resnet18.py
}

# ===========================================================================
# STEP9: Vitis AI Compile ResNet50 Imagenet Target Board
# ===========================================================================
compile_resnet50_imagenet(){
#train1
echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO STEP9] COMPILE IMAGENET QUANTIZED RESNET50"
echo "----------------------------------------------------------------------------------"
echo " "
source ./scripts/run_compile.sh zcu102  q_resnet50_imagenet.h5
source ./scripts/run_compile.sh vck190  q_resnet50_imagenet.h5
source ./scripts/run_compile.sh vek280  q_resnet50_imagenet.h5
source ./scripts/run_compile.sh vck5000 q_resnet50_imagenet.h5
source ./scripts/run_compile.sh v70     q_resnet50_imagenet.h5
mv   ./build/compiled_zcu102/zcu102_q_resnet50_imagenet.h5.xmodel  ./target/imagenet/zcu102_resnet50_imagenet.xmodel
mv   ./build/compiled_vck190/vck190_q_resnet50_imagenet.h5.xmodel  ./target/imagenet/vck190_resnet50_imagenet.xmodel
mv   ./build/compiled_vek280/vek280_q_resnet50_imagenet.h5.xmodel  ./target/imagenet/vek280_resnet50_imagenet.xmodel
mv ./build/compiled_vck5000/vck5000_q_resnet50_imagenet.h5.xmodel  ./target/imagenet/vck5000_resnet50_imagenet.xmodel
mv     ./build/compiled_v70/v70_q_resnet50_imagenet.h5.xmodel      ./target/imagenet/v70_resnet50_imagenet.xmodel
}

# ===========================================================================
# STEP10: Vitis AI Compile ResNet18 Imagenet Target Board
# ===========================================================================
compile_resnet18_imagenet(){
#train1
echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO STEP10] COMPILE IMAGENET QUANTIZED RESNET18"
echo "----------------------------------------------------------------------------------"
echo " "
source ./scripts/run_compile.sh zcu102  q_resnet18_imagenet.h5
source ./scripts/run_compile.sh vck190  q_resnet18_imagenet.h5
source ./scripts/run_compile.sh vek280  q_resnet18_imagenet.h5
source ./scripts/run_compile.sh vck5000 q_resnet18_imagenet.h5
source ./scripts/run_compile.sh v70     q_resnet18_imagenet.h5
mv   ./build/compiled_zcu102/zcu102_q_resnet18_imagenet.h5.xmodel  ./target/imagenet/zcu102_resnet18_imagenet.xmodel
mv   ./build/compiled_vck190/vck190_q_resnet18_imagenet.h5.xmodel  ./target/imagenet/vck190_resnet18_imagenet.xmodel
mv   ./build/compiled_vek280/vek280_q_resnet18_imagenet.h5.xmodel  ./target/imagenet/vek280_resnet18_imagenet.xmodel
mv ./build/compiled_vck5000/vck5000_q_resnet18_imagenet.h5.xmodel  ./target/imagenet/vck5000_resnet18_imagenet.xmodel
mv         ./build/compiled_v70/v70_q_resnet18_imagenet.h5.xmodel  ./target/imagenet/v70_resnet18_imagenet.xmodel
}


# ===========================================================================
# STEP11: prepare archive for TARGET runtime application for ImageNet
# ===========================================================================
prepare_imagenet_archives() {
echo " "
echo "----------------------------------------------------------------------------------"
echo "[DB INFO STEP11] PREPARING IMAGENET ARCHIVE FOR TARGET BOARDS"
echo "----------------------------------------------------------------------------------"
echo " "
if [ -d "./build/target" ]; then
  ### Terraset6 was running before this and you have to do nothing
  echo "./build/target exists already ..."
else
  ### Terraset6 was not called before this and you have to build some folders
  echo "./build/target does not exists ..."
  mkdir -p ./build/target
  mkdir -p ./build/target_vck190
  mkdir -p ./build/target_vck5000
  mkdir -p ./build/target_vek280
  mkdir -p ./build/target_zcu102
  mkdir -p ./build/target_v70
fi
cp -r ./target/imagenet ./build/target/
# zcu102
cp -r ./build/target/imagenet ./build/target_zcu102/
rm -f ./build/target_zcu102/imagenet/vck*_imagenet.xmodel
rm -f ./build/target_zcu102/imagenet/vek*_imagenet.xmodel
# vck190
cp -r ./build/target/imagenet ./build/target_vck190/
rm -f ./build/target_vck190/imagenet/zcu1*_imagenet.xmodel
rm -f ./build/target_vck190/imagenet/vek2*_imagenet.xmodel
rm -f ./build/target_vck190/imagenet/vck5*_imagenet.xmodel
# vek280
cp -r ./build/target/imagenet ./build/target_vek280/
rm -f ./build/target_vek280/imagenet/zcu*_imagenet.xmodel
rm -f ./build/target_vek280/imagenet/vck*_imagenet.xmodel
# vck5000
cp -r ./build/target/imagenet ./build/target_vck5000/
rm -f ./build/target_vck5000/imagenet/zcu1*_imagenet.xmodel
rm -f ./build/target_vck5000/imagenet/vek2*_imagenet.xmodel
rm -f ./build/target_vck5000/imagenet/vck1*_imagenet.xmodel
# v700
cp -r ./build/target/imagenet ./build/target_v70/
rm -f ./build/target_v70/imagenet/zcu1*_imagenet.xmodel
rm -f ./build/target_v70/imagenet/vek2*_imagenet.xmodel
rm -f ./build/target_v70/imagenet/vck*_imagenet.xmodel

# prerare tar files
cd ./build
tar -cvf  target_zcu102.tar  ./target_zcu102 > /dev/null
tar -cvf  target_vck190.tar  ./target_vck190 > /dev/null
tar -cvf  target_vek280.tar  ./target_vek280 > /dev/null
tar -cvf  target_vck5000.tar ./target_vck5000 > /dev/null
tar -cvf  target_v70.tar     ./target_v70 > /dev/null
#rm -r target
cd ..
}

# ===========================================================================
# remove imagenet test images
# ===========================================================================
remove_imagenet_test_images(){
  cd ./target/imagenet/
  rm -r ./val_dataset
  rm ./words.txt
  rm ./val.txt
  cd ../../
}

# ===========================================================================
# main for Terraset6
# ===========================================================================
# do not change the order of the following commands

main_terraset6(){
  echo " "
  echo " "
  pip install image-classifiers
  #terraset6_dataset            # 2
  #run_terraset6_training       # 3
  quantize_resnet18_terraset6   # 4
  compile_resnet18_terraset6    # 5
  ### if you want to cross compile the application for target from host side,
  ### which is not nexessary being compiled also on the target board,
  ### just uncomment next three lines
  #cd target
  #source ./terraset6/run_all_terraset6_target.sh compile_cif10
  #cd ..
  prepare_terraset6_archives    # 6
  echo " "
  echo " "
}

# ===========================================================================
# main for ImageNet
# ===========================================================================
# do not change the order of the following commands

main_imagenet(){
    echo " "
    echo "----------------------------------------------------------------------------------"
    echo "[DB INFO] NOW WORKING ON THE IMAGENET EXAMPLES"
    echo "----------------------------------------------------------------------------------"
    echo " "
    # patch for my code (do not touch it!)
    cp modelzoo/ImageNet/*.txt ./target/imagenet/

  prepare_imagenet_test_images
  ### uncomment next line if you are interested into ResNet50
  quantize_resnet50_imagenet    # 7
  quantize_resnet18_imagenet      # 8
  ### uncomment next line if you are interested into ResNet50
  compile_resnet50_imagenet     # 9
  compile_resnet18_imagenet       #10
  ### if you want to cross compile the application for target from host side,
  ### which is not nexessary being compiled also on the target board,
  ### just uncomment next three lines
  #cd target
  #source ./imagenet/run_all_imagenet_target.sh compile_resnet
  #cd ..
  remove_imagenet_test_images
  prepare_imagenet_archives
  echo " "
  echo " "
}


# ===========================================================================
# main for all
# ===========================================================================

# do not change the order of the following commands
main_all(){
  run_patch
  ### next line is commented: you should run it only once
  #run_clean_dos2unix  # step  1
  main_terraset6         # steps 2 to  6
  main_imagenet        # steps 7 to 11
}


# ===========================================================================
# DO NOT REMOVE THE FOLLOWING LINE
# ===========================================================================

"$@"
