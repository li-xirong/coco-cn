source ../common.ini
overwrite=0

for collection in flickr8ktrain flickr8kcntrain flickr30ktrain
do
    python build_vocab.py $collection --topk $vocab_size --overwrite $overwrite --rootpath $rootpath
done

