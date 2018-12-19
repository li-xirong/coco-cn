
# Step 1. prepare data 
dim=3
featurefile=toydata/FeatureData/f1/id.feature.txt
resultdir=toydata/FeatureData/f1
python txt2bin.py $dim $featurefile 0 $resultdir

# Step 2. search
python demo.py

