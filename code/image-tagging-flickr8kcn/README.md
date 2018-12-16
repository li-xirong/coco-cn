
# Cross-lingual image tagging

This package implements the following methods for training image tagging models that predict Chinese tags:

1. *Monolingual MLP*, which is trained using Chinese annotations of Flickr8k-CN.
2. *Multi-task MLP*, which is trained using bilingual (English-Chinese) annotations of Flickr8k-CN, where the English and Chinese losses are equally combined and minimized during training.
3. *Cascading MLP*, which is first trained on English annotations of Flickr30k to obtain semantic-enhanced image features and then trained using Chinese annotations of Flickr8k-CN.

## Requirements

* Ubuntu 16.04
* CUDA 9.0
* python 2.7
* TensorFlow 1.8
* numpy
* nltk


We used virtualenv to setup a deep learning workspace. Run the following script to install the required packages.
```shell
virtualenv --system-site-packages ~/deepws
source ~/deepws/bin/activate
pip install --upgrade pip
pip install tensorflow-gpu==1.8
pip install numpy
pip install nltk
deactivate
```

## Get started

### Data

```shell
mkdir $HOME/VisualSearch # this is set via rootpath in common.ini 
cd $HOME/VisualSearch
wget http://lixirong.net/data/coco-cn/image-tagging-flickr8kcn.tar.gz
tar xzf image-tagging-flickr8kcn.tar.gz
python check_data.py
```

### Scripts

Extract English and Chinese labels from sentences:
```shell
git clone https://github.com/li-xirong/coco-cn
cd code/image-tagging-flickr8kcn/preprocess
./do_extract_tags.sh flickr8k
./do_extract_tags.sh flickr30k
./do_extract_tags.sh flickr8kcn
./do_build_train_vocab.sh
cd ..
```

Train Chinese tagging models and evaluate them on the test set of Flickr8k-CN. 

```shell 
cd tf_tagging
./do_mono_zh_mlp.sh   
./do_multitask_mlp.sh
./do_cascade_mlp.sh
cd ..
```

### Performance on flickr8kcntest

The metrics are Precision, Recall and F-measure at top 5. 
Each metric is computed per image and averaged over all test images.

| Model | Precision | Recall | F-measure |
|:--- | ---:| ---:| ---:|
| Monolingual  MLP | 0.364 | 0.561 | 0.420 |
| Multi-task MLP   | 0.364 | 0.562 | 0.421 | 
| Cascading MLP    | 0.388 | 0.597 | 0.448 |

