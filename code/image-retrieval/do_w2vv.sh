
source common.ini

if [ $# -lt 2 ]; then
    echo "Usage: $0 language w2vv_config"
    exit 1
fi


lang=$1
model_config=$2

eval train_collection='$'"$lang"_train_collection
eval val_collection='$'"$lang"_val_collection
eval test_collection='$'"$lang"_test_collection

python w2vv_trainer.py $train_collection $val_collection $test_collection  --rootpath $rootpath --overwrite $overwrite --model_config $model_config

