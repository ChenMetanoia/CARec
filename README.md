# CollabAlginment

This this the anonymous repository for the KDD 2024 submitted paper **Collaborative Contextualization: Bridging the Gap between
Collaborative Filtering and Pre-trained Language Model**

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

### 3. Run CollabContex
```
python CollabContex.py
```
