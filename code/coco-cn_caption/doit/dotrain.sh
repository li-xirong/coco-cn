export CUDA_VISIBLE_DEVICES="0"

collection=mscoco-cn

model_name=fc
vf_name=pyresnext-101_rbps13k,flatten0_output,osl2
# multimodal feature for coco-cn+tag scheme
# vf_name=bow-top5+pyresnext-101_rbps13k,flatten0_output,osl2

batch_size=16
# num of workers reading feature data
num_workers=10

learning_rate=5e-4
learning_rate_decay_start=0
feedback_start_epoch=0

# set this to 1 for sequential learning
use_merged_vocab=0

num_epochs=80

python train.py --collection $collection \
                  --model_name $model_name --vf_name $vf_name \
                  --use_merged_vocab $use_merged_vocab \
                  --batch_size $batch_size --num_workers $num_workers\
                  --learning_rate $learning_rate \
                  --learning_rate_decay_start $learning_rate_decay_start \
                  --feedback_start_epoch $feedback_start_epoch \
                  --num_epochs $num_epochs \
                  --id expid

