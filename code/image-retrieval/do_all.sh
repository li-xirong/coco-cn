SECONDS=0

if [ $# -lt 1 ]; then
    echo "Usage: $0 w2vv_config"
    exit 1
fi

model_config=w2vv_attention_ctl
model_config=$1
fusion_config=zh2en2img_$model_config

# generate vocabulary
./do_prepare_bow.sh

# on original mscoco dataset
./do_w2vv.sh en $model_config

# on coco-cn dataset 
./do_w2vv.sh zh $model_config

# computing cross lingual similarity
./do_zh2en_match.sh $model_config

# compute cross lingual & modal simiarity
cd fusion
./do_search_fusion_weight.sh $fusion_config
cd ..

duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."

