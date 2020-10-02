count=1
while [$count -lt 300]
mv data/bench64_512/bench${count}binvoxmodel_normalized.solid.binvox data/bench64_256
var=1
count=`expr $count + $var`
done