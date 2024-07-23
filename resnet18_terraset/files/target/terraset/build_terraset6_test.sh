#!/bin/bash

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023


#cobblestonebrick dirtground grass pavement sand stairs 

tar -xvf test.tar &> /dev/null
#mv ./build/dataset/terraset6/test ./test
#rm -r ./build
cd ./test

cd cobblestonebrick
mv *.png ../
cd ..
rm -r cobblestonebrick/

cd dirtground
mv *.png ../
cd ..
rm -r dirtground/

cd grass
mv *.png ../
cd ..
rm -r grass/

cd pavement
mv *.png ../
cd ..
rm -r pavement/

cd sand
mv *.png ../
cd ..
rm -r sand/

cd stairs
mv *.png ../
cd ..
rm -r stairs/

cd ..
