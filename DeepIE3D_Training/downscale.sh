#!/bin/sh
output_dir="/home/ibalab/katayama/DeepIE3D/DeepIE3D_Training/data/bench64_512"
data_dir="/home/ibalab/katayama/DeepIE3D/DeepIE3D_Training/data/02828884/*"
dirs=`find $data_dir -maxdepth 0 -type d`
count=1
for dir in $dirs;
do
    echo $dir
    path="${dir}/models/"
    out_path="${output_dir}/bench${count}.binvox"
    python misc/create64voxels.py $path $out_path
    var=1
    count=`expr $count + $var`
done