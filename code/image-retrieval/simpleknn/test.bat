
::# Step 1. prepare data 
SET dim=3
SET featurefile=toydata/FeatureData/f1/id.feature.txt
SET resultdir=toydata/FeatureData/f1

python txt2bin.py %dim% %featurefile% 0 %resultdir%

::# Step 2. search
python demo.py

@pause

