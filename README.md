# CARec

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

Put meta dataset into ```data/Amazon/metadata``` directory. For example ```data/Amazon/metadata/meta_Electronics.json.gz```
Put reviews dataset into ```data/Amazon/metadata``` directory. For example ```data/Amazon/metadata/Electronics.json.gz```

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
python CollabContex.py
```
