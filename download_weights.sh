mkdir pretrained
wget -P pretrained http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz
mkdir pretrained/resnet_v2_50
tar zxvf pretrained/resnet_v2_50_2017_04_14.tar.gz -C pretrained/resnet_v2_50
rm pretrained/resnet_v2_50_2017_04_14.tar.gz
