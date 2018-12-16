cwd=`pwd`
rootpath=${cwd}/data
collection=$1

ln -s $rootpath/${collection}/FeatureData $rootpath/${collection}train/FeatureData
ln -s $rootpath/${collection}/FeatureData $rootpath/${collection}val/FeatureData
ln -s $rootpath/${collection}/FeatureData $rootpath/${collection}test/FeatureData
