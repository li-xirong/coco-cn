rootpath=@@@rootpath@@@
trainCollection=@@@trainCollection@@@
overwrite=@@@overwrite@@@

# similarity function
simi=cosine_batch

# model info
model_path=@@@model_path@@@
model_name=@@@model_name@@@
weight_name=@@@weight_name@@@

set_style=@@@set_style@@@

for testCollection in @@@testCollection@@@
do

python w2vv_tester.py  $trainCollection $testCollection  \
    --rootpath $rootpath \
    --simi $simi \
    --overwrite $overwrite \
    --model_path $model_path \
    --model_name $model_name \
    --weight_name $weight_name \
    --set_style $set_style
done

