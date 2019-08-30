# COCO-CN

COCO-CN is a bilingual image description dataset enriching MS-COCO with manually written Chinese sentences and tags. The new dataset can be used for multiple tasks including image tagging, captioning and retrieval, all in a cross-lingual setting. 


| Chinese sentences | COCO-CN train | COCO-CN val | COCO-CN  test| 
| -----:| -----:| -----:| -----:|
| human written    | :white_check_mark: | :white_check_mark: | :white_check_mark: | 
| human translation     | :x:     |   :x:  | :white_check_mark: | 
| machine translation (baidu)  | :white_check_mark: | :white_check_mark: | :white_check_mark: | 

<img src="dataset-snapshot.png" alt="coco-cn annotation examples"  width="400" />

## Progress

* version 201805: 20,341 images (training / validation / test: 18,341 / 1,000 /1,000), associated with 22,218 ***manually written*** Chinese sentences and 5,000 ***manually translated*** sentences. Data is freely available upon request. Please submit your request via [Google Form](https://goo.gl/forms/JMki8iD9OSvUAVWv1).
* [Precomputed image features](/data): ResNext-101
* [COCO-CN-Results-Viewer](https://github.com/evanmiltenburg/COCO-CN-Results-Viewer): A lightweight tool to inspect the results of different image captioning systems on the COCO-CN test set, developed by [Emiel van Miltenburg](https://emielvanmiltenburg.nl/) at the Tilburg University.
* [NUS-WIDE100](data/nuswide100): An extra test set.
+ 2018-12-16: Code for cross-lingual [image tagging](code/image-tagging-flickr8kcn) and [captioning](code/coco-cn_caption) released.
+ 2018-12-20: Code for [cross-lingual image retrieval](code/image-retrieval) and [our image annotation system](code/image-annotation-system) released.
+ 2019-01-13: The COCO-CN paper accepted as a regular paper by the T-MM journal.

## Citation

If you find COCO-CN useful, please consider citing the following paper:
* Xirong Li, Chaoxi Xu, Xiaoxu Wang, Weiyu Lan, Zhengxiong Jia, Gang Yang, Jieping Xu, [COCO-CN for Cross-Lingual Image Tagging, Captioning and Retrieval](https://arxiv.org/pdf/1805.08661.pdf), IEEE Transactions on Multimedia, Volume 21, Number 9, pages 2347-2360, 2019 
