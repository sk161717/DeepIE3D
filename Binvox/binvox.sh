#!/bin/sh
output_dir="/Users/katayamashunsuke/code/DeepIE3D/DeepIE3D_Training/data/bench"
this_dir="/Users/katayamashunsuke/code/DeepIE3D/Binvox/$1/*"
dirs=`find $this_dir -maxdepth 0 -type d`
count=1
cd $1
for dir in $dirs;
do
    echo $dir
    cd $dir
    path="${dir}/model.obj"
    out_path="$output_dir/bench${count}.binvox"
    binvox_path="${dir}/model.binvox"
    ../../binvox $path
    mv $binvox_path $out_path
    var=1
    count=`expr $count + $var`
done