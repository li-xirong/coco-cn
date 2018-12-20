rootpath=/home/xirong/VisualSearch
trainCollection=cococntrain
overwrite=0

# similarity function
simi=cosine_batch

# model info
model_path=/home/xirong/VisualSearch/cococntrain/cv_keras/cococnval/w2vv_trainer.py/w2vv_attention_ctl
model_name=model.json
weight_name=epoch_50.h5

set_style=ImageSets

for testCollection in cococnval cococntest
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


