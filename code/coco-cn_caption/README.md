
# Cross-lingual image captioning

This package contains source code for image captioning experiments described in our COCO-CN paper.

## Requirements

- Ubuntu 16.04
- CUDA 8.0
- Python 2.7
  
```
torch==0.4.0
torchvision==0.2.1
tqdm==4.23.4
matplotlib==2.2.2
nltk==3.3
numpy==1.14.3
scikit-image==0.13.1
scipy==1.1.0
```


## Get started 

### Data preparation

Get all needed text data and precomputed image features from the official repository of the [coco-cn](https://github.com/li-xirong/coco-cn) dataset.
  
First, create a folder named `mscoco-cn` in `data` and set the path of the dataset file as `data/mscoco-cn/TextData/dataset_mscoco-cn.json` and the corresponding annotation file as `data/mscoco-cn/TextData/captions_mscoco-cn.json`.

Then run the following commands: 
+ The `prepro_json.py` script will reset the sentence-ids in the dataset file into unique ids and split the dataset folder as well as the json dataset file into train-val-test subsets. 
+ The `build_vocab.py` script will generate a vocabulary file on the train split and dump it into `data/mscoco-cntrain/TextData/vocab_mscoco-cntrain.pkl`. 

```bash
$ python preprocess/prepro_json.py --collection mscoco-cn
$ python preprocess/build_vocab.py --collection mscoco-cn
```

For a fair comparison between different training shcemes, we use `merged_captions_coco-cn_test.json` as the standard test annotation file. Set its path to `data/mscoco-cntest/TextData/merged_captions_coco-cn_test.json` before starting any test procedure.

When the textdata is ready, you can put the downloaded feature into `data/mscoco-cn/FeatureData` and run the following command to make feature data available in the three splited dataset folder.

```bash
$ ./preprocess/prepro_feat.sh mscoco-cn
```

**Notice:** For experimentation on `mscoco-mixed`, `mscoco-mt`, prepare dataset folder with corresponding dataset file in the same way as the `mscoco-cn` example above.

## Train image captioning models

Run the following command to start training an image captioning model on coco-cn.

```bash
$ ./doit/dotrain.sh
```

The train script will dump checkpoints into `data/mscoco-cntrain/Models/`. 
Only the best-performing checkpoint on validation and the latest checkpoint is saved to save disk space.

You can configure other training schemes in `doit/dotrain.sh`.

## Evaluate trained model on the coco-cn test set

After the training process is done, you can run the follwing command to evaluate the trained model on the test split. 

**Notice:** Make sure the parameters are the same with that in `dotrain.sh`.

```bash
$ ./doit/dotest.sh
```

The `dotest.sh` script will
+ generate one predicted caption for each of the 1000 test images, and 
+ compute multiple performance scores including BLEU4, METEOR, ROUGE_L and CIDEr of these predicions using the [coco-caption](https://github.com/tylin/coco-caption) 
evaluation api.

## Other training schemes

### Sequential learning

To perform the `Sequential Learning` training scheme with `mscoco-mt` and `mscoco-cn`, put the `merged_vocab.pkl` file into `data/mscoco-mttrain/TextData` and  `data/mscoco-cntrain/TextData` where their original vocab file is replaced.

Then set the `use_merged_vocab` option to 1 in the doit scripts for sequential learning.

### Image Captioning with tags

When performing the `image captioning with tags` training scheme with coco-cn dataset, you need to temporarily set the `fc_feat_size` option in `opts.py` from 2,048 to 3,167, in order to fit the size of the provided multi-modal feature. Then configure the `vf_name` option in the doit scripts to exploit this scheme.

## Expected performance

The performance of the multiple training schemes on the coco-cn test set is as follows.

| Training scheme | BLEU4 | METEOR | ROUGE_L | CIDEr |
| -------- | -----|-------| ------- | ------- |
| coco-mt | 30.2  | 27.1 | 50.0 | 86.2  |
| coco-cn | 31.7  | 27.2 | 52.0 | 84.6 |
| coco-mixed | 29.8  | 28.6 | 50.3 | 86.8 |
| seqlearn | 36.7 | 29.5  | 55.0  | 98.4 |
| coco-cn+tag* | 31.3 | 30.1 | 53.2 | 90.0 |


## Developers

+ Weiyu Lan
+ Zhengxiong Jia
+ Hao Cheng

