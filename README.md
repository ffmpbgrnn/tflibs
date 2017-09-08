This repo contains source code for 
  - **Bidirectional Multirate Reconstruction for Temporal Modeling in Videos**, Linchao Zhu, Zhongwen Xu, Yi Yang.
  - **Few-Shot Object Recognition from Machine-Labeled Web Images**, Zhongwen Xu\*, Linchao Zhu\*, Yi Yang

To run our key-value memory experiments, you need to install TensorFlow 1.0 or above. The source code has been tested on TensorFlow 1.3.

```bash
# Download files (dataset, features, etc) for training
wget http://libstorage.pub/Expr/Mem/mem.zip -O /tmp/mem.zip
cd /tmp && unzip mem.zip

# Go to the source directory
# To train the model on GPU 0, run
./mem.sh train 0
# It will read the configure file configs/mem/few_shot_100.py

# To evaludate the model, run
./mem.sh eval /tmp/mem/few_shots/train_dir/{RUN_ID}/model.ckpt-${MODEL_ID} 1

Change the `train_stage` variable to `train_5` or `train_10` to evaluate the model on 5-shot and 10-shot settings.
```
