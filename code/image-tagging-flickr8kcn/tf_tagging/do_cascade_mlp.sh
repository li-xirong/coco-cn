source ../common.ini

zh_collection=flickr8kcn
en_collection=flickr30k

# train an English tagging model
./do_mono_mlp.sh $en_collection $vf_name

# use the English model to extract the semantic feature
./do_extract_enlabel_feat.sh $en_collection $zh_collection

# check if the semantic-enhanced feature is ready
label_feat_name=enlabel${vocab_size}_resnextl2_${en_collection}train
feat_dir=$rootpath/$zh_collection/FeatureData/${label_feat_name}+${vf_name}

if [[ ! -d "$feat_dir" ]]; then
    echo "$feat_dir not found"
    exit
fi

# train a Chinese tagging model using the semantic-enhanced features
./do_mono_mlp.sh $zh_collection ${label_feat_name}+${vf_name}

