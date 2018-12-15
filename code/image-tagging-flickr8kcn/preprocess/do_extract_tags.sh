
source ../common.ini

collection=flickr8k
collection=flickr30k


if [ $# -lt 1 ]; then
    echo "Usage: $0 collection (flickr8k, flickr30k, flickr8kcn)"
    exit 1
fi

collection=$1

if [[ $collection == "flickr8k" ]] || [[ $collection == "flickr30k" ]]; then
    lang=en
elif [[ $collection == "flickr8kcn" ]]; then
    lang=zh
else
    echo "invalid collection $collection"
    exit
fi

python extract_tags_from_sent.py $collection --lang $lang --rootpath $rootpath --overwrite $overwrite

src_file=$rootpath/$collection/TextData/$collection.imglabel.txt

if [[ ! -f $src_file ]]; then 
    echo "$src_file not found"
    exit
fi

for dataset in train val test
do
    subCollection=$collection$dataset
    imset_file=$rootpath/$subCollection/ImageSets/$subCollection.txt
    if [[ ! -f $imset_file ]]; then
        echo "$imset_file not found"
        continue
    fi
    res_file=$rootpath/$subCollection/TextData/$subCollection.imglabel.txt
    python get_subset.py $subCollection $src_file $res_file --overwrite $overwrite --rootpath $rootpath
done

