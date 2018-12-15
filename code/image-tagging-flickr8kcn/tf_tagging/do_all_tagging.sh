rootpath=$HOME/VisualSearch
train_collection=coco-cn_train
val_collection=coco-cn_val
test_collection=coco-cn_test
annotation_name=conceptslabel655.txt
en_train_collection=mscoco_train
en_annotation_name=engconceptslabel512.txt
vf_name=pyresnext-101_rbps13k,flatten0_output,osl2
model_name=baseline
overwrite=0
multi_task=0
top_k=655

if [ $# -lt 1 ]; then
    echo "Usage: $0 model(coco-mt, coco-cn, cascading, multi-task)"
    exit 1
fi

case $1 in
    coco-mt)
        echo 'coco-mt'
        train_collection=mscoco_train
        ;;
    coco-cn)
        echo 'coco-cn'
        ;;
    cascading)
        echo 'cascading'
        vf_name=pyresnext-101_rbps13k,flatten0_output,osl2+engconceptslabel512_mscoco_train_resnextl2
        ;;
    multi-task)
        echo 'multi-task'
        multi_task=1
        ;;
    *)
        echo 'invalid model'
        echo 'model must in (coco-mt,coco-cn, cascading, multi-task)'
        exit 1
        ;;
esac

python trainer.py --train_collection $train_collection --val_collection $val_collection --annotation_name $annotation_name \
    --model_name $model_name --vf_name $vf_name --rootpath $rootpath --overwrite $overwrite --multi_task $multi_task \
    --aux_train_collection $en_train_collection --aux_annotation_name $en_annotation_name 
    
python apply.py --test_collection $test_collection --train_collection $train_collection --val_collection $val_collection \
    --annotation_name $annotation_name --model_name $model_name --vf_name $vf_name --rootpath $rootpath --top_k $top_k \
    --overwrite $overwrite --multi_task $multi_task --aux_train_collection $en_train_collection \
    --aux_annotation_name $en_annotation_name

python eval.py --rootpath $rootpath --train_collection $train_collection --val_collection $val_collection \
    --test_collection $test_collection --model_name $model_name --annotation_name $annotation_name \
    --multi_task $multi_task --aux_train_collection $en_train_collection --aux_annotation_name $en_annotation_name \
    --vf_name $vf_name --overwrite $overwrite 
