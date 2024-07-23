#!/bin/bash

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  10 Aug. 2023


TARGET=$2
#vek280


#clean
clean_terraset6(){
echo " "
echo "clean terraset6"
echo " "
cd terraset6
rm -rf test
rm -f *~
rm -f  run_cnn cnn* get_dpu_fps *.txt
rm -rf rpt
rm -f  *.txt
rm -f  *.log
mkdir -p rpt
cd ..
}

# compile CNN application
compile_terraset6(){
echo " "
echo "compile terraset6"
echo " "
cd terraset6/code
echo "PWD1 = " $PWD
bash -x ./build_app.sh
mv code ../cnn_resnet18_terraset6 # change name of the application
bash -x ./build_get_dpu_fps.sh
mv code ../get_dpu_fps
cd ../..
echo "PWD2 = " $PWD
}

# build cifar10 test images
test_images_terraset6(){
echo " "
echo "build test images for terraset6"
echo " "
cd terraset6
bash ./build_terraset6_test.sh
cd ..
echo " "
echo "PWD3 = " $PWD
}

# now run the terraset6 classification with 4 CNNs using VART C++ APIs
run_cnn_terraset6(){
echo " "
echo " run terraset6 CNN"
echo " "
cd terraset6
./cnn_resnet18_terraset6 ./${TARGET}_train1_resnet18_terraset6.xmodel ./test/ ./terraset6_labels.dat | tee ./rpt/predictions_terraset6_resnet18.log
# check DPU prediction accuracy
bash -x ./terraset6_performance.sh ${TARGET}
echo "PWD4 = " $PWD
cd ..
}

#remove images
end_terraset6(){
echo " "
echo "end of terraset6"
echo " "
cd terraset6
rm -rf test
cd ../
echo "PWD5 = " $PWD
#tar -cvf target.tar ./target_*
}


main()
{
    clean_terraset6
    compile_terraset6
    test_images_terraset6
    run_cnn_terraset6
    end_terraset6
}




"$@"
