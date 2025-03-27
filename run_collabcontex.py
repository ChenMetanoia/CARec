import argparse
import logging
import torch
from recbole.config import Config
from recbole.data.utils import create_dataset
from recbole.utils import init_seed, ensure_dir
from recbole.trainer import Trainer
from model.collabcontex import CollabContex
from utils.logger import init_logger
from utils.util import data_preparation, ColdEvaluator

def set_up_config(args):
    config_dict = vars(args)
    if args.PLM in ('instructor-xl', 'bert-base-uncased', 'bge-base-en-v1.5'):
        item_embedding_dim = 768
    elif args.PLM in ('all-MiniLM-L6-v2', 'all-mpnet-base-v2'):
        item_embedding_dim = 384
    else:
        raise NotImplementedError

    config_dict['mlp'] = {
        'input_dim': item_embedding_dim,
        'hidden_dims': [item_embedding_dim, item_embedding_dim],
        'output_dim': item_embedding_dim,
        'num_layers': 2,
        'dropout': 0.2
    }
    
    config_dict['embedding_size'] = item_embedding_dim
    config_dict['data_path'] = f'dataset/{args.PLM}'
    
    return config_dict

def initialize_model_and_trainer(config_dict):
    config_dict["checkpoint_dir"] = f"results/{config_dict['PLM']}/{config_dict['dataset']}/CollabFusion/"
    
    config_file_list = ['properties/train.yaml', 'properties/data.yaml']
    config = Config(model=CollabContex,
                    config_dict=config_dict,
                    config_file_list=config_file_list)
    
    init_seed(config['seed'], config['reproducibility'])
    ensure_dir(config['checkpoint_dir'])
    
    init_logger(config)
    logger = logging.getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    
    if 'cold' in config['benchmark_filename']:
        train_data, valid_data, test_data, cold_data = data_preparation(config, dataset)
    else:
        train_data, valid_data, test_data = data_preparation(config, dataset)
    
    model = CollabContex(config, train_data.dataset).to(config['device'])
    logger.info(model)
    
    trainer = Trainer(config, model)
    
    return trainer, config, logger, train_data, valid_data, test_data, cold_data

def run_evaluation_phases(trainer, config, logger, train_data, valid_data, test_data, cold_data):
    # Item Tutoring Phase
    print('Item Tutoring Phase')
    trainer.model.set_item_tutoring_phase()
    trainer.cur_step = 0 # reset cur_step
    best_valid_score, _ = trainer.fit(train_data, valid_data, show_progress=False)
    
    # load best model
    checkpoint = torch.load(trainer.saved_model_file, map_location=trainer.device)
    trainer.model.load_state_dict(checkpoint["state_dict"])
    trainer.model.load_other_parameter(checkpoint.get("other_parameter"))
    print('load model from', trainer.saved_model_file)
    
    # User Tutoring Phase
    print('User Tutoring Phase')
    trainer.model.set_user_tutoring_phase()
    trainer.cur_step = 0 # reset cur_step
    best_valid_score, _ = trainer.fit(train_data, valid_data, show_progress=False)
    
    # Evaluation
    # Warm
    trainer.model.encoder.not_apply_mlp = False
    test_result = trainer.evaluate(test_data)
    logger.info(f'warm results: {test_result}')
    
    # Cold
    if 'cold' in config['benchmark_filename']:
        trainer.model.encoder.not_apply_mlp = True
        cold_evaluator = ColdEvaluator(trainer)
        cold_test_result = cold_evaluator.evaluate_cold_item(cold_data, comparewith='item_emb', agg_neighbor=False)
        logger.info(f'cold results: {cold_test_result}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, help='GPU ID')
    parser.add_argument('--dataset', default='Electronics', type=str, help='[Electronics, Office_Products, Grocery_and_Gourmet_Food]')
    parser.add_argument('--num_layers', default=2, type=int, help='number of layers')
    parser.add_argument('--PLM', default='all-mpnet-base-v2', help='[all-mpnet-base-v2, all-MiniLM-L6-v2, bert-base-uncased, bge-base-en-v1.5, instructor-xl]')
    args = parser.parse_args()
    
    config_dict = set_up_config(args)
    
    trainer, config, logger, train_data, valid_data, test_data, cold_data = initialize_model_and_trainer(config_dict)
    
    run_evaluation_phases(trainer, config, logger, train_data, valid_data, test_data, cold_data)
