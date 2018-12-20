# Cross-lingual image retrieval

We release source code of enhanced Word2VisualVec (W2VV), which is used for cross-lingual image retrieval described in our COCO-CN paper. 
In particular, we enhance the standard [W2VV](https://github.com/danieljf24/w2vv) model by
1. adding an attention layer after GRU to take into account all the hidden vectors, and 
2. substituting the original mean squared error (MSE) for the contrastive loss (CTL).


## Requirements

* Ubuntu 16.04
* CUDA 9.0
* python 2.7.12
* gensim 3.6.0
* TensorFlow 1.11.0 
* keras 2.2.4 with TensorFlow backend
* pydot 1.2.4 for keras model visualization

We used virtualenv to setup a deep learning workspace that supports keras with TensorFlow backend.
Run the following script to install the required packages.
```shell
virtualenv --system-site-packages ~/deepws
source ~/deepws/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
deactivate
```

## Get started

### Data

| Required | Size | Description |
|:--- |:--- |:--- |
| word2vec.tar.gz | 3.1G | English word2vec trained on Flickr tags |
| zh_w2v.zip | 2.1G | Chinese word2vec trained on Chinese wiki documents | 
| coco-cn_retrieval_data.tar.gz | 894M | coco-cn and mscoco data for retrieval |

```shell
mkdir $HOME/VisualSearch # this is set via rootpath in common.ini 
cd $HOME/VisualSearch

# download and extract pre-trained word2vec models
wget http://lixirong.net/data/w2vv-tmm2018/word2vec.tar.gz # English word2vec
wget http://lixirong.net/data/coco-cn/zh_w2v.zip # Chinese word2vec
tar xzf word2vec.tar.gz
unzip zh_w2v.zip

# download coco-cn data
wget http://lixirong.net/data/coco-cn/coco-cn_retrieval_data.tar.gz
tar xzf coco-cn_retrieval_data.tar.gz
```

| Dataset | Images |  English / Chinese Sentences | 
|:--- | ---:| ---:|
mscoco2014dev-train             | 121,286 | 606,757 |
cococntrain                     |  18,341 |  20,021 |
mscoco2014dev-val / cococnval   |   1,000 |   1,000 |
mscoco2014dev-test / cococntest |   1,000 |   1,000 |

### Scripts

Train and evaluate a standard w2vv model:

```shell
./do_all.sh base_w2vv
```

Train and evaluate an enhanced w2vv model:

```shell
./do_all.sh w2vv_attention_ctl
```

### Performance on the COCO-CN test set

Notice that as we upgraded the deep learning environment (keras from 1.0 to 2.2), the performance of both w2vv and enhanced w2vv models is better than those reported in our COCO-CN paper. 

| Model | CL | CM | CLM |
|:--- | ---:| ---:| ---:|
| w2vv | 0.470 | 0.418 | 0.498 |
| enhanced w2vv | 0.481 | 0.432 | 0.514 | 


## Developers

+ Chaoxi Xu
+ Jianfeng Dong
+ Xirong Li
