import yaml
import os
import pandas as pd
from utils import RawDataProcessor

if __name__ == '__main__':
    # data_name = 'Grocery_and_Gourmet_Food'
    # data_name = 'Office_Products'
    data_name = 'Electronics'
    with open(f'data_process_config_amazon_{data_name}.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print(config)
    data_processor = RawDataProcessor(config)

    if os.path.exists(f'item_meta_data_{data_name}.csv'):
        data_processor.item_meta_data = pd.read_csv(f'item_meta_data_{data_name}.csv', sep='\t')
    else:
        data_processor.load_meta_data()
        data_processor.item_meta_data.to_csv(f'item_meta_data_{data_name}.csv', index=False, sep='\t')
    if os.path.exists(f'inter_data_{data_name}.csv'):
        data_processor.inter_data = pd.read_csv(f'inter_data_{data_name}.csv', sep='\t')
    else:
        data_processor.load_inter_data()
        data_processor.inter_data.to_csv(f'inter_data_{data_name}.csv', index=False, sep='\t')
    data_processor.preprocess()
    # remove the item_meta_data_{data_name}.csv and inter_data_{data_name}.csv
    os.remove(f'item_meta_data_{data_name}.csv')
    os.remove(f'inter_data_{data_name}.csv')