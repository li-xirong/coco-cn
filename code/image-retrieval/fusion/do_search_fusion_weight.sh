
source ../common.ini

searchCollection=$zh_val_collection
testCollection=$zh_test_collection
zh_trainCollection=$zh_train_collection

if [ $# -lt 1 ]; then
    echo "Usage: $0 fusion_config"
    exit 1
fi

config=$1

python search_fusion_weight.py $searchCollection $testCollection $config --zh_trainCollection $zh_trainCollection --rootpath $rootpath #--overwrite $overwrite

python gather_eval_res.py $testCollection $config --zh_trainCollection $zh_trainCollection --rootpath $rootpath
