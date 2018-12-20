# generate the word vocabulary of the training set

source common.ini
overwrite=0

for collection in $zh_train_collection $en_train_collection
do
    caption_file=$rootpath/$collection/TextData/$collection.caption.txt
    if [ ! -f "$caption_file" ]; then
        echo "caption file $caption_file not found"
        continue
    fi

    for text_style in bow bow_filterstop
    do
        python build_vocab.py $collection $text_style $freq_threshold --overwrite $overwrite --rootpath $rootpath
    done
done

