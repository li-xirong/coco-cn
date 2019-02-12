
## Precomputed image features

We use pre-trained CNN models to extract visual features from images. The features are stored in a binary format, which can be read by [BigFile](https://github.com/li-xirong/jingwei/blob/master/util/simpleknn/bigfile.py), see our [wiki](https://github.com/li-xirong/coco-cn/wiki) page for detailed instruction. 

1. [2048-dim ResNext-101 feature (145 MB)](http://lixirong.net/data/coco-cn/coco-cn_resnext-101_feat.tar.gz). 
```bash
wget http://lixirong.net/data/coco-cn/coco-cn_resnext-101_feat.tar.gz
```



## Supporting data

* [NUS-WIDE100](nuswide100): A set of 100 images randomly selected from the NUS-WIDE dataset for user study.
* [Chinese tag vocabulary](conceptscn655.txt): A set of 655 Chinese tags defined for the cross-lingual image tagging task.
* [Sentences with typos](detected-typos.txt): A list of sentences with typos detected thus far. Although we tried our best to collect high-quality annotations, small typos are unfortunately inevitable.

## Acknowledgments

* We thank the [MediaMill](https://ivi.fnwi.uva.nl/isis/mediamill/) team at the University of Amsterdam for generously providing their trained ResNext-101 model. 
* We thank Miss Xinru Chen for performing typos check on COCO-CN sentences.
