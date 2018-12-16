export CUDA_VISIBLE_DEVICES="0"

collection=mscoco-cn

model_name=fc
vf_name=pyresnext-101_rbps13k,flatten0_output,osl2
# multimodal feature for `image captioning with tags`
# vf_name=bow-top5+pyresnext-101_rbps13k,flatten0_output,osl2

batch_size=64
# num of workers reading feature data
num_workers=5

learning_rate=5e-4
feedback_start_epoch=0

# set this to 1 for sequential learning
use_merged_vocab=0

beam_size=3

# set language_eval to 0 to not evaluate the test results
# when there is no groundtruth for test set
language_eval=1

python test.py --collection $collection \
                 --model_name $model_name --vf_name $vf_name \
                 --use_merged_vocab $use_merged_vocab \
                 --batch_size $batch_size --num_workers $num_workers\
                 --learning_rate $learning_rate \
                 --feedback_start_epoch $feedback_start_epoch \
                 --beam_size $beam_size \
                 --language_eval $language_eval \
                 --id expid

