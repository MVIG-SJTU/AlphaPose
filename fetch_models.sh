#get res152
wget mvig.sjtu.edu.cn/publications/rmpe/output.zip
unzip -d output.zip human-detection
rm output.zip

#get model
mkdir -p predict/models
wget mvig.sjtu.edu.cn/publications/rmpe/final_model.t7 -P predict/models
