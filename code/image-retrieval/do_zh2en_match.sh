source common.ini

if [ $# -lt 1 ]; then
    echo "Usage: $0 w2vv_config"
    exit 1
fi

w2vv_config=$1

for style in val test
do

zh_testCollection=cococn${style}
en_testCollection=mscoco2014dev-${style}

zh_feat_dir=$rootpath/$zh_testCollection/SimilarityIndex/$zh_train_collection/$w2vv_config
en_feat_dir=$rootpath/$en_testCollection/SimilarityIndex/$en_train_collection/$w2vv_config

for feat_dir in $zh_feat_dir $en_feat_dir 
do
    if [[ ! -d "$feat_dir" ]]; then
        echo $feat_dir not found
        exit
    fi
done

python zh2en_sim.py $zh_testCollection $en_testCollection $w2vv_config --zh_trainCollection $zh_train_collection --en_trainCollection $en_train_collection --rootpath $rootpath --overwrite $overwrite

done
