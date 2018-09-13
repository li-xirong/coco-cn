
## Evaluation on Chinese image captioning

* [coco-cn-cap-eval.tar.gz ](http://lixirong.net/data/coco-cn/coco-cn-cap-eval.tar.gz)(109M). 
This package provides data and code that re-produce Table 6 of [our arxiv paper](https://arxiv.org/abs/1805.08661). 

```bash
python eval_cncap.py

====================
Training Bleu_4 METEOR ROUGE_L CIDEr
flickr8k-cn 10.1 14.9 33.8 22.9
aic-icc 7.4 21.3 34.2 24.6
coco-mt 30.2 27.1 50.0 86.2
coco-cn 31.7 27.2 52.0 84.6
seqlearn 36.7 29.5 55.0 98.4
```
