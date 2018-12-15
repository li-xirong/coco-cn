source ../common.ini


if [ $# -lt 2 ]; then
    echo "Usage: $0 collection feature"
    exit 1
fi


collection=$1
vf_name=$2
train_collection=${collection}train
annotation_name=concepts${train_collection}${vocab_size}.txt
concept_file=$rootpath/$train_collection/Annotations/$annotation_name
val_collection=${collection}val
test_collection=${collection}test

python trainer.py --train_collection $train_collection --val_collection $val_collection --annotation_name $annotation_name --model_name $model_name --vf_name $vf_name --rootpath $rootpath --overwrite $overwrite --multi_task 0 --aux_train_collection dummpy --aux_annotation_name dummpy

python apply.py --test_collection $test_collection --train_collection $train_collection --val_collection $val_collection --annotation_name $annotation_name --model_name $model_name --vf_name $vf_name --rootpath $rootpath --top_k $top_k --overwrite $overwrite --multi_task 0 --aux_train_collection dummpy --aux_annotation_name dummpy

tagging_method=$annotation_name/$train_collection/$val_collection/$model_name/$vf_name
python eval.py $test_collection $concept_file $tagging_method --rootpath $rootpath
