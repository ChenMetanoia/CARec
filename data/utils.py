import os
import re
import json
import gzip
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import html
import torch
import torch.nn.functional as F
import jsonlines


class RawDataProcessor:
    """ Raw data processor for [Amazon, Yelp, MovieLens] dataset into
    RecBole data format.
    
    Args:
        config (dict): Configuration dictionary containing the following parameters:
            - 'data_source': Source of the dataset, must be one of ['amazon', 'yelp', 'ml-1m'].
            - 'data_name': Name of the dataset.
            - 'output_dir': Directory where output data will be saved.
            - 'seed': Random seed for reproducibility.
            - 'task': Task name, must be one of ['general', 'seq'].
            - 'plm': Pre-trained language model for text embeddings.
            - 'inter_data_col': Interaction data column names.
            - 'inter_sample_ratio': Ratio of interaction data to sample.
            - 'eval_args' (dict, optional): Evaluation settings.
            - 'input_dir_dict' (dict, optional): Input data directories for user, item, and interaction data.
            - 'input_data_name_dict' (dict, optional): Input data names for user, item, and interaction data.
            - 'input_file_suffix' (str, optional): Suffix for input data files.
            - 'gpu_id' (int, optional): GPU ID to use.
            - 'user_meta_data_col' (str, optional): User meta-data column name.
            - 'item_meta_data_col' (str, optional): Item meta-data column name.
            - 'user_k_core_threshold' (int, optional): User k-core threshold.
            - 'item_k_core_threshold' (int, optional): Item k-core threshold.
            - 'text_feature_list' (list, optional): List of text features.
            - 'cold_ratio' (float, optional): Ratio for cold-start recommendation.
    """
    
    def __init__(self, config, **kwargs):
        # Essential settings
        self.data_source = config.get('data_source')
        self.data_name = config.get('data_name')
        self.output_dir = os.path.join(config.get('output_dir'), config['plm'])
        self.seed = config.get('seed')
        self.task = config['task']
        self.plm = config['plm']
        self.inter_data_col = config['inter_data_col']
        self.inter_sample_ratio = config['inter_sample_ratio']

        # Optional settings with defaults
        self.eval_args = config.get('eval_args')
        self.input_dir_dict = config.get('input_dir_dict', {})
        self.input_data_name_dict = config.get('input_data_name_dict', {})
        self.input_file_suffix = config.get('input_file_suffix', '')
        self.gpu_id = config.get('gpu_id', -1)
        self.user_meta_data_col = config.get('user_meta_data_col')
        self.item_meta_data_col = config.get('item_meta_data_col')
        self.user_k_core_threshold = config.get('user_k_core_threshold', 3)
        self.item_k_core_threshold = config.get('item_k_core_threshold', 3)
        self.text_feature_list = config.get('text_feature_list', [])
        self.cold_ratio = config.get('cold_ratio', 0)

        # Data placeholders
        self.user_meta_data = None  # intermediate data
        self.item_meta_data = None  # intermediate data
        self.user_data = None  # output data
        self.item_data = None  # output data
        self.inter_data = None  # output data
        self.cold_inter_data = None  # output data

        # Check output directory
        check_path(self.output_dir)
        
        
    def load_meta_data(self):
        # Load raw meta data from input_path_dict into unified dataFrame.
        print('Load meta data...')
        for meta_name in self.input_data_name_dict:
            if meta_name == 'user':
                self.user_meta_data = self._load_user_meta_data(os.path.join(self.input_dir_dict['user'], 
                                                                             self.input_data_name_dict['user'])+self.input_file_suffix['user'])
            elif meta_name == 'item':
                self.item_meta_data = self._load_item_meta_data(os.path.join(self.input_dir_dict['item'], 
                                                                             self.input_data_name_dict['item']+self.input_file_suffix['item']))
            elif meta_name == 'inter':
                continue
            else:
                raise NotImplementedError('Meta data name {} is not supported'.format(meta_name))
            
    def load_inter_data(self):
        # Load raw interaction data from input_path_dict into unified dataFrame.
        print('Load interaction data...')
        self.inter_data = self._load_inter_data(os.path.join(self.input_dir_dict['inter'], 
                                                             self.input_data_name_dict['inter']+self.input_file_suffix['inter']))

    def preprocess(self):
        # Preprocess meta data and interaction data.
        print('Preprocess inter data...')
        self._preprocess_inter_data()
        print('Preprocess meta data...')
        self._preprocess_meta_data()
        print('Generate embedding...')
        self.generate_embedding()
        print('Map data into index...')
        self._map_id_to_index(self.inter_data, self.item_data, self.cold_inter_data)
        self.save_data()
        print('Done!')
        
    def save_data(self):
        # make sure that output_dir exists
        check_path(os.path.join(self.output_dir, self.data_name))
        # Save preprocessed data into output_path_dict.
        print('Save meta data...')
        if self.user_data is not None:
            self._save_user_meta_data(os.path.join(self.output_dir, self.data_name, self.data_name+'.user'))
        if self.item_data is not None:
            self._save_item_meta_data(os.path.join(self.output_dir, self.data_name, self.data_name+'.item'))
        if self.inter_data is not None:
            split_ratio = list(self.eval_args['split'].values())[0]
            train, valid, test = split_data(self.inter_data, split_ratio, self.seed, self.eval_args['group_by'])
            print('Save inter data...')
            train.to_csv(os.path.join(self.output_dir, self.data_name, self.data_name+'.train.inter'), sep='\t', index=False)
            valid.to_csv(os.path.join(self.output_dir, self.data_name, self.data_name+'.valid.inter'), sep='\t', index=False)
            test.to_csv(os.path.join(self.output_dir, self.data_name, self.data_name+'.test.inter'), sep='\t', index=False)
            if self.cold_ratio > 0:
                self.cold_inter_data.to_csv(os.path.join(self.output_dir, self.data_name, self.data_name+'.cold.inter'), sep='\t', index=False)
            
    def generate_embedding(self):
        # Generate embedding for meta data.
        if self.item_meta_data is not None:
            # split item_meta_data into item_id:token list and text list
            item_id_list = self.item_meta_data['item_id:token'].tolist()
            text_list = self.item_meta_data['text'].tolist()
            self.item_data = self._generate_text_embedding(item_id_list, text_list)
        if self.user_meta_data is not None:
            raise NotImplementedError('generate_embedding for user_meta_data is not implemented')
        
    def _map_id_to_index(self, inter_df, embeddings_df, cold_inter_df):
        # Create item and user mapping dictionaries
        item_map_dict = dict(zip(embeddings_df['item_id:token'].unique(), range(len(embeddings_df))))
        user_map_dict = dict(zip(inter_df['user_id:token'].unique(), range(len(inter_df['user_id:token'].unique()))))

        # Map item and user IDs to their corresponding indices
        inter_df['item_id:token'] = inter_df['item_id:token'].map(item_map_dict).astype(int)
        inter_df['user_id:token'] = inter_df['user_id:token'].map(user_map_dict).astype(int)

        cold_inter_df['item_id:token'] = cold_inter_df['item_id:token'].map(item_map_dict).astype(int)
        cold_inter_df['user_id:token'] = cold_inter_df['user_id:token'].map(user_map_dict).astype(int)

        embeddings_df['item_id:token'] = embeddings_df['item_id:token'].map(item_map_dict).astype(int)

        return inter_df, embeddings_df, cold_inter_df
        
    def _save_user_meta_data(self, output_path):
        raise NotImplementedError('save_user_meta_data is not implemented')
    
    def _save_item_meta_data(self, output_path):
        self.item_data.to_csv(output_path, sep='\t', index=False)
        
    def _generate_text_embedding(self, items, texts):
        if self.plm == 'instructor-xl' or self.plm == 'all-MiniLM-L6-v2' or self.plm=='all-mpnet-base-v2':
            if self.plm == 'instructor-xl':
                from InstructorEmbedding import INSTRUCTOR
                model = INSTRUCTOR('hkunlp/instructor-xl')
            elif self.plm == 'all-MiniLM-L6-v2':
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-MiniLM-L6-v2')
            elif self.plm == 'all-mpnet-base-v2':
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-mpnet-base-v2')
            else:
                raise NotImplementedError('plm {} is not supported'.format(self.plm))
            
            if self.plm == 'instructor-xl':
                instruction = "Represent the Amazon title: "
                sentence = []
                for t in texts:
                    sentence.append(instruction + t) 
            else:
                sentence = texts
            embeddings = model.encode(sentence, 
                                      batch_size=32, 
                                      show_progress_bar=True,
                                      convert_to_numpy=True,
                                      device='cuda:2',
            )
        else:
            device = set_device(self.gpu_id)
            tokenizer, model = load_plm(self.plm)
            model = model.to(device)
            model.eval()  # Set the model to evaluation mode
            
            embeddings = []
            batch_size = 4
            
            with torch.no_grad():  # Disable gradient computation for inference
                if self.plm == 'bert-base-uncased':
                    for i in tqdm(range(0, len(texts), batch_size), desc='Generating Embeddings', unit='batches'):
                        batch_texts = texts[i:i+batch_size]
                        encoded_batch = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
                        encoded_batch = encoded_batch.to(device)
                        outputs = model(**encoded_batch)
                        cls_output = outputs.last_hidden_state[:, 0, ].cpu().tolist()
                        embeddings.extend(cls_output)
                elif self.plm == 'bge-base-en-v1.5':
                    for i in tqdm(range(0, len(texts), batch_size), desc='Generating Embeddings', unit='batches'):
                        batch_texts = texts[i:i+batch_size]
                        encoded_batch = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
                        encoded_batch = encoded_batch.to(device)
                        model_output = model(**encoded_batch)
                        # Perform pooling. In this case, cls pooling.
                        sentence_embeddings = model_output[0][:, 0]
                        embeddings.extend(sentence_embeddings)
                    # normalize embeddings
                    embeddings = torch.stack(embeddings, dim=0)
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Convert embeddings to a DataFrame
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
        elif isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().tolist()
        elif isinstance(embeddings, list):
            pass
        else:
            raise NotImplementedError('embeddings type {} is not supported'.format(type(embeddings)))
        embeddings_df = pd.DataFrame({"item_id:token": items, "item_emb:float_seq": embeddings})
        
        # Convert embeddings to a string representation
        embeddings_df['item_emb:float_seq'] = embeddings_df['item_emb:float_seq'].apply(lambda x: ' '.join(map(str, x)))
        
        return embeddings_df
    
    def _preprocess_inter_data(self):
        # Remove duplicates based on 'item_id:token'
        self.item_meta_data = self.item_meta_data.drop_duplicates(subset='item_id:token')
        # Sample inter_data based on the sample ratio
        self.inter_data = self.inter_data.sample(frac=self.inter_sample_ratio, random_state=self.seed)
        self.inter_data = self.k_core(self.inter_data, self.user_k_core_threshold, self.item_k_core_threshold)
        
        if self.item_meta_data is not None:
            # Filter item_meta_data based on the items present in inter_data
            item_set = set(self.inter_data['item_id:token'])
            self.item_meta_data = self.item_meta_data[self.item_meta_data['item_id:token'].isin(item_set)]
            # Make sure that item_set has the same size as item_meta_data
            item_set = set(self.item_meta_data['item_id:token'])
            assert len(item_set) == len(self.item_meta_data)
            # Make sure that item_set is a subset of inter_data
            self.inter_data = self.inter_data[self.inter_data['item_id:token'].isin(item_set)]
            
            if self.cold_ratio > 0:
                # Extract the cold set from item_meta_data
                cold_set = self.extract_cold_set(self.item_meta_data, self.cold_ratio, seed=self.seed)
                print('    The number of cold items: %d' % len(cold_set))
                
                # Filter cold items from inter_data and create cold_inter_data
                self.cold_inter_data = self.inter_data[self.inter_data['item_id:token'].isin(cold_set)]
                print('    The number of cold inters: %d' % len(self.cold_inter_data))
                self.inter_data = self.inter_data[~self.inter_data['item_id:token'].isin(cold_set)]
                print('    The number of inters: %d' % len(self.inter_data))
                
                # Apply k-core to the remaining inter_data
                self.inter_data = self.k_core(self.inter_data, self.user_k_core_threshold, self.item_k_core_threshold)

                # Filter users in cold_inter_data that are not in inter_data
                user_set = set(self.inter_data['user_id:token'])
                self.cold_inter_data = self.cold_inter_data[self.cold_inter_data['user_id:token'].isin(user_set)]
                
                # Combine item sets from inter_data and cold_inter_data
                item_set = set(self.inter_data['item_id:token']) | set(self.cold_inter_data['item_id:token'])
            
                # Update item_meta_data to contain all items in the combined item set
                self.item_meta_data = self.item_meta_data[self.item_meta_data['item_id:token'].isin(item_set)]
                print('    The number of items: %d' % len(self.item_meta_data))
                
                # Assertions
                assert len(self.cold_inter_data) > 0
                assert len(set(self.inter_data['item_id:token']) & set(self.cold_inter_data['item_id:token'])) == 0
                assert len(set(self.item_meta_data['item_id:token'])) == len(set(self.inter_data['item_id:token']) | set(self.cold_inter_data['item_id:token']))
                
        # Sort interaction data based on evaluation arguments
        if self.eval_args['order'] == 'TO':
            if self.eval_args['group_by'] == 'user':
                # Sort by user_id and timestamp
                self.inter_data = self.inter_data.sort_values(by=['user_id:token', 'timestamp:float'])
                if self.cold_ratio > 0:
                    self.cold_inter_data = self.cold_inter_data.sort_values(by=['user_id:token', 'timestamp:float'])
            else:
                # Sort by timestamp
                self.inter_data = self.inter_data.sort_values(by=['timestamp:float'])
                if self.cold_ratio > 0:
                    self.cold_inter_data = self.cold_inter_data.sort_values(by=['timestamp:float'])
        elif self.eval_args['order'] == 'RO':
            # Randomly shuffle the data
            self.inter_data = self.inter_data.sample(frac=1, random_state=self.seed)
            if self.cold_ratio > 0:
                self.cold_inter_data = self.cold_inter_data.sample(frac=1, random_state=self.seed)

        
    def _preprocess_meta_data(self):
        if self.user_meta_data is not None:
            self._preprocess_user_meta_data()
        if self.item_meta_data is not None:
            self._preprocess_item_meta_data()
            
    def extract_cold_set(self, df, cold_ratio, seed):
        # Extract cold set from interaction data.
        df = df.sample(frac=1, random_state=seed)
        cold_set_size = int(len(df) * cold_ratio)
        cold_set = set(df.iloc[:cold_set_size]['item_id:token'])
        return cold_set
    
    def k_core(self, df, user_k_core_threshold, item_k_core_threshold):
        # Filter out users and items that have less than k interactions.
        idx = 0
        while True:
            user_counts = df['user_id:token'].value_counts()
            item_counts = df['item_id:token'].value_counts()
            if (user_counts < user_k_core_threshold).sum() == 0 and (item_counts < item_k_core_threshold).sum() == 0:
                break
            df = df[df['user_id:token'].isin(user_counts[user_counts >= user_k_core_threshold].index)]
            df = df[df['item_id:token'].isin(item_counts[item_counts >= item_k_core_threshold].index)]
            idx += 1
            print('    Epoch %d The number of inters: %d, users: %d, items: %d'
                    % (idx, len(df), len(user_counts), len(item_counts)))
        return df
        
    def _preprocess_user_meta_data(self):
        raise NotImplementedError('preprocess_user_meta_data is not implemented')
    
    def _preprocess_item_meta_data(self):
        if len(self.text_feature_list) > 0:
            self.item_meta_data = self._preprocess_text_feature(self.item_meta_data, self.data_source)
            
    def _preprocess_text_feature(self, meta_df, data_source):
        if data_source == 'amazon':
            # Function to concatenate title, brand, and category
            def concatenate_info(row):
                # Convert the list in the 'category' column to a string
                try:
                    category_str = ' '.join(eval(row['category']))
                except TypeError:
                    category_str = ' '.join(row['category'])
                # remove the category feaute which has the same value as the dataset name
                category_str = category_str.replace(amazon_dataset2categoryname[self.data_name], '')
                # Concatenate title, brand, and category into one sentence
                sentence = f"{row['title']} {row['brand']} {category_str}"
                return sentence

            # concat text features from text_feature_list
            meta_df['text'] = meta_df.apply(concatenate_info, axis=1)
            # clean text
            meta_df['text'] = meta_df['text'].apply(lambda x: clean_text(x))
            # drop text_feature_list columns
            meta_df = meta_df.drop(columns=self.text_feature_list)
        else:
            raise NotImplementedError('preprocess_text_feature for {} is not implemented'.format(data_source))
        return meta_df
        
    def _load_user_meta_data(self, input_path):
        raise NotImplementedError('load_user_meta_data is not implemented')
    
    def _load_item_meta_data(self, input_path):
        file_format = check_file_format(input_path, ['json', 'json.gz', 'csv', 'dat', 'text', 'text.gz'])
        df = self._load_raw_data_into_dataframe(input_path, self.item_meta_data_col, file_format)
        return df

    def _load_inter_data(self, input_path):
        file_format = check_file_format(input_path, ['json', 'json.gz', 'csv', 'dat', 'text', 'text.gz'])
        df = self._load_raw_data_into_dataframe(input_path, self.inter_data_col, file_format)
        return df
        
    def _load_raw_data_into_dataframe(self, input_path, column_names, file_format):
        columns_to_keep = list(column_names.keys())
        data_dict = {col: [] for col in columns_to_keep}
        
        if file_format == 'json':
            with open(input_path, 'r') as f:
                for line in tqdm(f, desc='Loading JSON', unit=' lines'):
                    data = json.loads(line)
                    for col in columns_to_keep:
                        if col in data:
                            data_dict[col].append(data[col])
        
        elif file_format == 'json.gz':
            with gzip.open(input_path, 'rt') as f:
                with jsonlines.Reader(f) as reader:
                    for data in tqdm(reader, desc='Loading JSON', unit=' lines'):
                        for col in columns_to_keep:
                            if col in data:
                                data_dict[col].append(data[col])
        
        elif file_format == 'csv':
            with open(input_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in tqdm(reader, desc='Loading CSV', unit=' rows'):
                    for col in columns_to_keep:
                        if col in row:
                            data_dict[col].append(row[col])
        
        elif file_format == 'dat' or file_format == 'text':
            with open(input_path, 'r') as f:
                for line in tqdm(f, desc='Loading DAT/TEXT', unit=' lines'):
                    data = line.strip().split()
                    if len(data) == len(columns_to_keep):
                        for col, value in zip(columns_to_keep, data):
                            data_dict[col].append(value)
        
        elif file_format == 'text.gz':
            with gzip.open(input_path, 'rt') as f:
                for line in tqdm(f, desc='Loading TEXT.GZ', unit=' lines'):
                    data = line.strip().split()
                    if len(data) == len(columns_to_keep):
                        for col, value in zip(columns_to_keep, data):
                            data_dict[col].append(value)
        
        else:
            raise NotImplementedError('File format {} is not supported'.format(file_format))
        
        # Create a Pandas DataFrame
        df = pd.DataFrame(data_dict).rename(columns=column_names)
        
        return df
            

def load_json(path, features, file_format='json'):
    data_dict = {key: [] for key in features}
    
    if file_format == 'json.gz':
        open_file = gzip.open
    else:
        open_file = open

    with open_file(path, 'r') as fp:
        for line in tqdm(fp, desc='Generate text'):
            data = json.loads(line)
            for meta_key in features:
                if meta_key in data:
                    data_dict[meta_key].append(data[meta_key])

    df = pd.DataFrame(data_dict)
    return df

def split_data(df, split_ratio, seed, group_by_name=None):
    if group_by_name is not None:
        grouped_ratings = df.groupby(group_by_name)
        
        train_data = []
        valid_data = []
        test_data = []

        for reviewerID, group in tqdm(grouped_ratings, total=len(grouped_ratings), desc='Split data:'):
            # Shuffle the group
            group = group.sample(frac=1, random_state=seed)
            
            # Calculate the split points
            n = len(group)
            train_idx = int(split_ratio[0] * n)
            valid_idx = int(split_ratio[1] * n) + train_idx
            
            # Split the group into train, valid, and test sets
            train_set = group.iloc[:train_idx]
            valid_set = group.iloc[train_idx:valid_idx]
            test_set = group.iloc[valid_idx:]
            
            # Append the sets to the corresponding lists
            train_data.append(train_set)
            valid_data.append(valid_set)
            test_data.append(test_set)

        # Concatenate the data lists into DataFrames
        train_df = pd.concat(train_data)
        valid_df = pd.concat(valid_data)
        test_df = pd.concat(test_data)
        
        return train_df, valid_df, test_df
        
    else:
        # Split data into train set, valid set and test set.
        df = df.sample(frac=1, random_state=seed)
        train_ratio = split_ratio[0]
        valid_ratio = split_ratio[1]
        train_size = int(len(df) * train_ratio)
        valid_size = int(len(df) * valid_ratio)
        train_set = df.iloc[:train_size]
        valid_set = df.iloc[train_size: train_size + valid_size]
        test_set = df.iloc[train_size + valid_size:]
        return train_set, valid_set, test_set
    
def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def set_device(gpu_id):
    if gpu_id == -1:
        return torch.device('cpu')
    else:
        return torch.device(
            'cuda:' + str(gpu_id) if torch.cuda.is_available() else 'cpu')

def load_plm(model_name='bert-base-uncased'):
    if model_name == 'bert-base-uncased':
        model_name = model_name
    elif model_name == 'all-MiniLM-L6-v2' or model_name == 'all-mpnet-base-v2':
        model_name = f'sentence-transformers/{model_name}'
    elif model_name == 'bge-base-en-v1.5':
        model_name = 'BAAI/bge-base-en-v1.5'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

def check_file_format(file_path, file_format_list):
    if not os.path.exists(file_path):
        raise ValueError('File path {} does not exist'.format(file_path))
    file_format = file_path.split('.')[-1]
    # check second order file format
    if file_format == 'gz':
        file_format = file_path.split('.')[-2] + '.' + file_format
    if file_format not in file_format_list:
        raise ValueError('File format {} is not supported'.format(file_format))
    return file_format

def clean_text(raw_text):
    if isinstance(raw_text, list):
        cleaned_text = ' '.join(raw_text)
    elif isinstance(raw_text, dict):
        cleaned_text = str(raw_text)
    else:
        cleaned_text = raw_text
    cleaned_text = html.unescape(cleaned_text)
    cleaned_text = re.sub(r'["\n\r]*', '', cleaned_text)
    index = -1
    while -index < len(cleaned_text) and cleaned_text[index] == '.':
        index -= 1
    index += 1
    if index == 0:
        cleaned_text = cleaned_text + '.'
    else:
        cleaned_text = cleaned_text[:index] + '.'
    if len(cleaned_text) >= 2000:
        cleaned_text = ''
    return cleaned_text
    
amazon_dataset2fullname = {
    'Beauty': 'All_Beauty',
    'Fashion': 'AMAZON_FASHION',
    'Appliances': 'Appliances',
    'Arts': 'Arts_Crafts_and_Sewing',
    'Automotive': 'Automotive',
    'Books': 'Books',
    'CDs': 'CDs_and_Vinyl',
    'Cell': 'Cell_Phones_and_Accessories',
    'Clothing': 'Clothing_Shoes_and_Jewelry',
    'Music': 'Digital_Music',
    'Electronics': 'Electronics',
    'Gift': 'Gift_Cards',
    'Food': 'Grocery_and_Gourmet_Food',
    'Home': 'Home_and_Kitchen',
    'Scientific': 'Industrial_and_Scientific',
    'Kindle': 'Kindle_Store',
    'Luxury': 'Luxury_Beauty',
    'Magazine': 'Magazine_Subscriptions',
    'Movies': 'Movies_and_TV',
    'Instruments': 'Musical_Instruments',
    'Office': 'Office_Products',
    'Garden': 'Patio_Lawn_and_Garden',
    'Pantry': 'Prime_Pantry',
    'Pet': 'Pet_Supplies',
    'Software': 'Software',
    'Sports': 'Sports_and_Outdoors',
    'Tools': 'Tools_and_Home_Improvement',
    'Toys': 'Toys_and_Games',
    'Games': 'Video_Games'
}

amazon_dataset2categoryname = {
    'Grocery_and_Gourmet_Food' : 'Grocery & Gourmet Food',
    'Electronics': 'Electronics',
    'Office_Products': 'Office_Products'
}