
source ../common.ini

en_collection=flickr30k
test_collection=flickr8kcn

if [ $# -lt 2 ]; then
    echo "Usage: $0 en_collection test_collection"
    exit 1
fi
en_collection=$1
test_collection=$2
en_train_collection=${en_collection}train
en_val_collection=${en_collection}val
en_annotation_name=concepts${en_train_collection}${vocab_size}.txt

concept_file=$rootpath/$en_train_collection/Annotations/$en_annotation_name
tagging_method=$en_annotation_name/$en_train_collection/$en_val_collection/baseline/$vf_name
label_feat_name=enlabel${vocab_size}_resnextl2_${en_train_collection}

python apply.py --test_collection $test_collection --train_collection $en_train_collection --val_collection $en_val_collection --annotation_name $en_annotation_name --model_name $model_name --vf_name $vf_name --rootpath $rootpath --top_k $top_k  --overwrite $overwrite --multi_task $multi_task --aux_train_collection $aux_train_collection --aux_annotation_name $aux_annotation_name


python tagvotes2bin.py $test_collection $concept_file $tagging_method ${label_feat_name} --overwrite $overwrite --rootpath $rootpath

label_feat_dir=$rootpath/$test_collection/FeatureData/${label_feat_name}

if [[ ! -d "$label_feat_dir" ]]; then
    echo "$label_feat_dir not found"
    exit
fi

python concat_label_visual_feat.py $test_collection "$label_feat_name" $vf_name --rootpath $rootpath

#python simpleknn/norm_feat.py $label_feat_dir --overwrite $overwrite --rootpath $rootpath
#python concat_label_visual_feat.py $test_collection "$label_feat_name"l2 $vf_name --rootpath $rootpath

