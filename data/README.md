
## Precomputed image features

We use pre-trained CNN models to extract visual features from images. The features are stored in a binary format, which can be read by [BigFile](https://github.com/li-xirong/jingwei/blob/master/util/simpleknn/bigfile.py), see our [wiki](https://github.com/li-xirong/coco-cn/wiki) page for detailed instruction. 

1. [2048-dim ResNext-101 feature (145 MB)](http://lixirong.net/data/coco-cn/coco-cn_resnext-101_feat.tar.gz) 
```bash
wget http://lixirong.net/data/coco-cn/coco-cn_resnext-101_feat.tar.gz
```

## Acknowledgments

We thank the [MediaMill](https://ivi.fnwi.uva.nl/isis/mediamill/) team at the University of Amsterdam for generously providing their trained ResNext-101 model. 
