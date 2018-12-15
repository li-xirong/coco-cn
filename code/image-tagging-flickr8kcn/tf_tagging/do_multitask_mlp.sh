source ../common.ini

collection=flickr8kcn
en_train_collection=flickr8ktrain

train_collection=${collection}train
val_collection=${collection}val
test_collection=${collection}test

annotation_name=concepts${train_collection}${vocab_size}.txt
concept_file=$rootpath/$train_collection/Annotations/$annotation_name

en_annotation_name=concepts${en_train_collection}${vocab_size}.txt

python trainer.py --train_collection $train_collection --val_collection $val_collection --annotation_name $annotation_name --model_name $model_name --vf_name $vf_name --rootpath $rootpath --overwrite $overwrite --multi_task 1 --aux_train_collection $en_train_collection --aux_annotation_name $en_annotation_name 

  
python apply.py --test_collection $test_collection --train_collection $train_collection --val_collection $val_collection --annotation_name $annotation_name --model_name $model_name --vf_name $vf_name --rootpath $rootpath --top_k $top_k --overwrite $overwrite --multi_task 1 --aux_train_collection $en_train_collection --aux_annotation_name $en_annotation_name

tagging_method=$annotation_name/$en_annotation_name/$train_collection/$en_train_collection/$val_collection/$model_name/$vf_name

python eval.py $test_collection $concept_file $tagging_method --rootpath $rootpath

