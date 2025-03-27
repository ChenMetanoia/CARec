# CARec

This this the repository for the paper [**Collaborative Alignment for Recommendation**](https://dl.acm.org/doi/abs/10.1145/3627673.3679535)

## Requirements

```
recbole==1.1.1
python==3.8.5
cudatoolkit==11.3.1
pytorch==1.12.1
pandas==1.3.0
transformers==4.18.0
```

### 1. Download [Amazon meta dataset](https://nijianmo.github.io/amazon/index.html) and [Amazon reviews dataset](https://nijianmo.github.io/amazon/index.html)
Category: Electronics, Office Products, Grocery and Gourmet Food

Data: metadata

Put meta dataset into ```data/Amazon/Metadata``` directory. For example ```data/Amazon/Metadata/meta_Electronics.json.gz```
Put reviews dataset into ```data/Amazon/Reviews``` directory. For example ```data/Amazon/Reviews/Electronics.json.gz```

### 2. Process data
```
cd data
python process_data.py
```

### 3. Run CARec
```
python CollabContex.py
```


## Requirements

```
recbole==1.1.1
python==3.8.5
cudatoolkit==11.3.1
pytorch==1.12.1
pandas==1.3.0
transformers==4.18.0
```

### 1. Download [Amazon meta dataset](https://nijianmo.github.io/amazon/index.html) and [Amazon reviews dataset](https://nijianmo.github.io/amazon/index.html)
Category: Electronics, Office Products, Grocery and Gourmet Food

Data: metadata

Put meta dataset into ```data/Amazon/metadata``` directory. For example ```data/Amazon/metadata/meta_Electronics.json.gz```
Put reviews dataset into ```data/Amazon/metadata``` directory. For example ```data/Amazon/metadata/Electronics.json.gz```

### 2. Process data
```
cd data
python process_data.py
```

### 3. Run CARec
```
python run_collabcontex.py
```

#### 3.1 Using Specific GPU to run
For example, using gpu 1 instead of 0.
```
CUDA_VISIBLE_DEVICES=1 python run_collabcontex.py
```

#### 3.2 ⚠️ If You See This Error When Loading Checkpoints:
```
RuntimeError: Attempting to deserialize object on CUDA device 0 but torch.cuda.device_count() is 0.
Please use torch.load with map_location to map your storages to an existing device.
```
Please go to:
```
recbole/trainer/trainer.py
```
And replace:
```
checkpoint = torch.load(resume_file, map_location=self.device)
```
With:
```
checkpoint = torch.load(resume_file, map_location=lambda storage, loc: storage.cuda() if torch.cuda.is_available() else storage)
```
This ensures compatibility when loading checkpoints across different devices or environments.

### 4. If you think this paper is useful please consider cite:
```
@inproceedings{wang2024collaborative,
  title={Collaborative Alignment for Recommendation},
  author={Wang, Chen and Yang, Liangwei and Liu, Zhiwei and Liu, Xiaolong and Yang, Mingdai and Liang, Yueqing and Yu, Philip S},
  booktitle={Proceedings of the 33rd ACM International Conference on Information and Knowledge Management},
  pages={2315--2325},
  year={2024}
}
```
