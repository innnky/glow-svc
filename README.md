# Glow-SVC

another implement by rcell based on official [glow-tts](https://github.com/jaywalnut310/glow-tts) repo
+ content-vec + fcpe(f0) -> glowtts -> vocos -> wav
+ 坑多多多多多
+ fp16会炸炸炸

pretrain:
+ [content-vec-best](https://huggingface.co/lengyue233/content-vec-best/resolve/main/pytorch_model.bin)
+ [fcpe](https://huggingface.co/datasets/ylzz1997/rmvpe_pretrain_model/resolve/main/fcpe.pt)
+ [vocos](https://huggingface.co/charactr/vocos-mel-24khz/resolve/main/pytorch_model.bin)


preprocess：
+ [resample.py](resample.py) -> [preprocess_flist_config.py](preprocess_flist_config.py) 
-> [extract_vec.py](extract_vec.py) -> [extract_f0_mel.py](extract_f0_mel.py)

train:
+ python train.py -c configs/config.json -m model_name

infer:
+ 暂无