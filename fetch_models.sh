cd human-detection
wget mvig.sjtu.edu.cn/publications/rmpe/output.zip
unzip output.zip
rm output.zip
cd ../predict
mkdir models
cd models
wget mvig.sjtu.edu.cn/publications/rmpe/final_model.t7
cd ../..

