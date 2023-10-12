import importlib
import torch
import pickle
from recbole.data.dataloader import *
from recbole.sampler import KGSampler, Sampler, RepeatableSampler
from recbole.utils import ModelType, set_color


def save_tensor_to_file(file_path, tensor):
    # Open the file in append mode
    with open(file_path, 'ab') as file:
        # Save the tensor to the file
        torch.save(tensor, file)
    
def empty_file(file_path):
    # Open the file in write mode
    with open(file_path, 'w'):
        pass  # Do nothing, just need to create the file
      
def import_class(model_name):
    """Import module from module path.

    Args:
        module_path (str): Module path.

    Returns:
        module: Imported module.

    """
    model_root_path = 'recbole_gnn.model.general_recommender'
    if model_name == 'LightGCN':
        module_path = '.'.join([model_root_path, model_name.lower()])
    else:
        module_path = '.'.join([model_root_path, 'basic_gnn'])
    module = importlib.import_module(module_path)
    cls = getattr(module, model_name)
    return cls


def data_preparation(config, dataset):
    """Split the dataset by :attr:`config['[valid|test]_eval_args']` and create training, validation and test dataloader.

    Note:
        If we can load split dataloaders by :meth:`load_split_dataloaders`, we will not create new split dataloaders.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset): An instance object of Dataset, which contains all interaction records.

    Returns:
        tuple:
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    model_type = config["MODEL_TYPE"]
    built_datasets = dataset.build()

    datasets = built_datasets
    if len(datasets) == 4:
        train_dataset, valid_dataset, test_dataset, cold_dataset = built_datasets
        train_sampler, valid_sampler, test_sampler, cold_sampler = create_samplers(
            config, dataset, built_datasets, cold=True
        )
    else:
        train_dataset, valid_dataset, test_dataset = built_datasets
        train_sampler, valid_sampler, test_sampler = create_samplers(
            config, dataset, built_datasets
        )

    if model_type != ModelType.KNOWLEDGE:
        train_data = get_dataloader(config, "train")(
            config, train_dataset, train_sampler, shuffle=config["shuffle"]
        )
    else:
        kg_sampler = KGSampler(
            dataset,
            config["train_neg_sample_args"]["distribution"],
            config["train_neg_sample_args"]["alpha"],
        )
        train_data = get_dataloader(config, "train")(
            config, train_dataset, train_sampler, kg_sampler, shuffle=True
        )

    valid_data = get_dataloader(config, "valid")(
        config, valid_dataset, valid_sampler, shuffle=False
    )
    test_data = get_dataloader(config, "test")(
        config, test_dataset, test_sampler, shuffle=False
    )
    if len(datasets) == 4:
        cold_data = get_dataloader(config, "test")(
            config, cold_dataset, cold_sampler, shuffle=False
        )
    if config["save_dataloaders"]:
        if len(datasets) == 4:
            save_split_dataloaders(
                config, dataloaders=(train_data, valid_data, test_data, cold_data)
            )
        else:
            save_split_dataloaders(
                config, dataloaders=(train_data, valid_data, test_data)
            )

    logger = getLogger()
    logger.info(
        set_color("[Training]: ", "pink")
        + set_color("train_batch_size", "cyan")
        + " = "
        + set_color(f'[{config["train_batch_size"]}]', "yellow")
        + set_color(" train_neg_sample_args", "cyan")
        + ": "
        + set_color(f'[{config["train_neg_sample_args"]}]', "yellow")
    )
    logger.info(
        set_color("[Evaluation]: ", "pink")
        + set_color("eval_batch_size", "cyan")
        + " = "
        + set_color(f'[{config["eval_batch_size"]}]', "yellow")
        + set_color(" eval_args", "cyan")
        + ": "
        + set_color(f'[{config["eval_args"]}]', "yellow")
    )
    if len(datasets) == 4:
        return train_data, valid_data, test_data, cold_data
    else:
        return train_data, valid_data, test_data


def get_dataloader(config, phase):
    """Return a dataloader class according to :attr:`config` and :attr:`phase`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        phase (str): The stage of dataloader. It can only take two values: 'train' or 'evaluation'.

    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`phase`.
    """
    register_table = {
        "MultiDAE": _get_AE_dataloader,
        "MultiVAE": _get_AE_dataloader,
        "MacridVAE": _get_AE_dataloader,
        "CDAE": _get_AE_dataloader,
        "ENMF": _get_AE_dataloader,
        "RaCT": _get_AE_dataloader,
        "RecVAE": _get_AE_dataloader,
    }

    if config["model"] in register_table:
        return register_table[config["model"]](config, phase)

    model_type = config["MODEL_TYPE"]
    if phase == "train":
        if model_type != ModelType.KNOWLEDGE:
            return TrainDataLoader
        else:
            return KnowledgeBasedDataLoader
    else:
        eval_mode = config["eval_args"]["mode"][phase]
        if eval_mode == "full":
            return FullSortEvalDataLoader
        else:
            return NegSampleEvalDataLoader


def _get_AE_dataloader(config, phase):
    """Customized function for VAE models to get correct dataloader class.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        phase (str): The stage of dataloader. It can only take two values: 'train' or 'evaluation'.

    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`phase`.
    """
    if phase == "train":
        return UserDataLoader
    else:
        eval_mode = config["eval_args"]["mode"]
        if eval_mode == "full":
            return FullSortEvalDataLoader
        else:
            return NegSampleEvalDataLoader

def _create_sampler(
    dataset,
    built_datasets,
    distribution: str,
    repeatable: bool,
    alpha: float = 1.0,
    base_sampler=None,
):
    phases = ["train", "valid", "test"]
    sampler = None
    if distribution != "none":
        if base_sampler is not None:
            base_sampler.set_distribution(distribution)
            return base_sampler
        if not repeatable:
            sampler = Sampler(
                phases,
                built_datasets,
                distribution,
                alpha,
            )
        else:
            sampler = RepeatableSampler(
                phases,
                dataset,
                distribution,
                alpha,
            )
    return sampler

def create_samplers(config, dataset, built_datasets, cold=False):
    """Create sampler for training, validation, testing, and cold.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset): An instance object of Dataset, which contains all interaction records.
        built_datasets (list of Dataset): A list of split Dataset, which contains dataset for
            training, validation and testing.
        cold (bool, optional): Whether to create sampler for cold. Defaults to False.

    Returns:
        tuple:
            - train_sampler (AbstractSampler): The sampler for training.
            - valid_sampler (AbstractSampler): The sampler for validation.
            - test_sampler (AbstractSampler): The sampler for testing.
            - cold_sampler (AbstractSampler): The sampler for cold.
    """
    phases = ["train", "valid", "test"]
    train_neg_sample_args = config["train_neg_sample_args"]
    valid_neg_sample_args = config["valid_neg_sample_args"]
    test_neg_sample_args = config["test_neg_sample_args"]
    repeatable = config["repeatable"]
    base_sampler = _create_sampler(
        dataset,
        built_datasets,
        train_neg_sample_args["distribution"],
        repeatable,
        train_neg_sample_args["alpha"],
    )
    train_sampler = base_sampler.set_phase("train") if base_sampler else None

    valid_sampler = _create_sampler(
        dataset,
        built_datasets,
        valid_neg_sample_args["distribution"],
        repeatable,
        base_sampler=base_sampler,
    )
    valid_sampler = valid_sampler.set_phase("valid") if valid_sampler else None

    test_sampler = _create_sampler(
        dataset,
        built_datasets,
        test_neg_sample_args["distribution"],
        repeatable,
        base_sampler=base_sampler,
    )
    test_sampler = test_sampler.set_phase("test") if test_sampler else None
    
    if cold:
        cold_sampler = _create_sampler(
            dataset,
            built_datasets,
            test_neg_sample_args["distribution"],
            repeatable,
            base_sampler=base_sampler,
        )
        cold_sampler = cold_sampler.set_phase("test") if cold_sampler else None
        return train_sampler, valid_sampler, test_sampler, cold_sampler
    return train_sampler, valid_sampler, test_sampler

def get_user_emb(dataset):
    user_id_list = dataset.inter_feat['user_id'].tolist()
    item_id_list = dataset.inter_feat['item_id'].tolist()
    item_emb = dataset.item_feat.item_emb
    
    cur_uid = user_id_list[0]
    last_uid = user_id_list[-1]
    cur_uid_emb = item_emb[item_id_list[0]]
    count = 1
    user_emb = torch.zeros(dataset.user_num, dataset.item_feat.item_emb.shape[1])
    for idx in range(1, len(user_id_list)):
        if cur_uid == user_id_list[idx] and idx != len(user_id_list) - 1:
            cur_uid_emb += item_emb[item_id_list[idx]]
            count += 1
        elif idx == len(user_id_list) - 1:
            cur_uid_emb += item_emb[item_id_list[idx]]
            user_emb[cur_uid] = cur_uid_emb / count
            user_emb[user_id_list[idx]] = item_emb[item_id_list[idx]]
        else:
            user_emb[cur_uid] = cur_uid_emb / count
            cur_uid = user_id_list[idx]
            cur_uid_emb = item_emb[item_id_list[idx]]
            count = 1
    return user_emb

def generate_item_kmeans_emb(dataset, config):
    import os
    import torch
    import faiss
    # item centroid initialization
    n_components = round(config['cluster_percentage'] * dataset.item_num)
    cluster = KMeans(
        num_cluster=n_components,
        seed=2023,
        hidden_size=config['embedding_size'],
        niter=config['niter'],
        gpu_id=config['gpu_id'],
        device=config["device"],
    )
    item_emb = dataset.item_feat.item_emb[1:]
    cluster.train(item_emb) # exclude padding item
    cls, _ = cluster.query(item_emb)
    cls = cls.cpu()
    item_avg_emb = generate_avg_item_emb(cls, item_emb)
    # Create a row of zerosc
    zeros_row = torch.zeros(1, item_avg_emb.size(1), dtype=item_avg_emb.dtype)
    # Concatenate the zeros to the top of the tensor
    item_avg_emb = torch.cat((zeros_row, item_avg_emb))
    path = os.path.join(config['data_path'], f"item_centroid_avgemb_{config['cluster_percentage']}_{config['niter']}.pt")
    torch.save(item_avg_emb, path)
    print(f"save item centroid emb to {path}")
    return item_avg_emb

def generate_avg_item_emb(cls, item_emb):
    index_tensor = cls
    item_embedding_tensor = item_emb
    
    # Sort the index tensor and item embedding tensor based on the index tensor
    sorted_indices = torch.argsort(index_tensor)
    sorted_index_tensor = index_tensor[sorted_indices]
    sorted_item_embedding_tensor = item_embedding_tensor[sorted_indices]

    # Create a mapping from original values to contiguous values
    unique_values = torch.unique(sorted_index_tensor)
    contiguous_values = torch.arange(len(unique_values))
    value_map = {value.item(): contiguous_value.item() for value, contiguous_value in zip(unique_values, contiguous_values)}

    # Map the tensor values to their contiguous counterparts
    sorted_index_tensor = torch.tensor([value_map[value.item()] for value in sorted_index_tensor])

    # recover the order of the mapped values
    index_tensor = sorted_index_tensor[sorted_indices.argsort()]

    # Calculate the average item embedding based on the same index
    unique_indices, unique_inverse_indices = torch.unique(sorted_index_tensor, return_inverse=True)
    average_item_embedding = torch.zeros(len(unique_indices), item_embedding_tensor.shape[1])

    for i, index in enumerate(unique_indices):
        mask = unique_inverse_indices == i
        group_embeddings = sorted_item_embedding_tensor[mask]
        average_embedding = torch.mean(group_embeddings, dim=0)
        average_item_embedding[i] = average_embedding
        
    # Repeat the average embeddings for each occurrence of the corresponding index
    repeated_indices = torch.repeat_interleave(unique_indices, torch.bincount(index_tensor))
    repeated_item_embedding = average_item_embedding[repeated_indices]

    # Recover the original order using argsort on the sorted indices
    original_order = torch.argsort(sorted_indices)

    # Apply the original order to the average item embeddings
    recovered_average_item_embedding = repeated_item_embedding[original_order]
    return recovered_average_item_embedding
    
    
def save_split_dataloaders(config, dataloaders):
    """Save split dataloaders.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataloaders (tuple of AbstractDataLoader): The split dataloaders.
    """
    # ensure_dir(config["checkpoint_dir"])
    # save_path = config["checkpoint_dir"]
    # saved_dataloaders_file = f'{config["dataset"]}-for-{config["model"]}-dataloader.pth'
    file_path = config['dataloaders_save_path']
    logger = getLogger()
    logger.info(set_color("Saving split dataloaders into", "pink") + f": [{file_path}]")
    Serialization_dataloaders = []
    for dataloader in dataloaders:
        generator_state = dataloader.generator.get_state()
        dataloader.generator = None
        dataloader.sampler.generator = None
        Serialization_dataloaders += [(dataloader, generator_state)]

    with open(file_path, "wb") as f:
        pickle.dump(Serialization_dataloaders, f)
        
    
def evaluate_cold_item(
    trainer, eval_data, load_best_model=True, model_file=None, show_progress=False
    ):
    # generate mask for cold item
    id_tensor = eval_data.dataset.inter_feat['item_id'].unique()
    mask = torch.ones(eval_data.dataset.item_num).bool()
    mask[id_tensor] = False
    if load_best_model:
        checkpoint_file = model_file or trainer.saved_model_file
        checkpoint = torch.load(checkpoint_file, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint["state_dict"])
        trainer.model.load_other_parameter(checkpoint.get("other_parameter"))
        message_output = "Loading model structure and parameters from {}".format(
            checkpoint_file
        )
        trainer.logger.info(message_output)
        
    trainer.model.eval()
    
    iter_data = (
        tqdm(
            eval_data,
            total=len(eval_data),
            ncols=100,
            desc=set_color(f"Evaluate   ", "pink"),
        )
        if show_progress
        else eval_data
    )

    num_sample = 0
    for batch_idx, batched_data in enumerate(iter_data):
        num_sample += len(batched_data)
        interaction, scores, positive_u, positive_i = eval_cold_func(trainer, batched_data, mask)
        if trainer.gpu_available and show_progress:
            iter_data.set_postfix_str(
                set_color("GPU RAM: " + get_gpu_usage(trainer.device), "yellow")
            )
        trainer.eval_collector.eval_batch_collect(
            scores, interaction, positive_u, positive_i
        )
    trainer.eval_collector.model_collect(trainer.model)
    struct = trainer.eval_collector.get_data_struct()
    result = trainer.evaluator.evaluate(struct)
    if not trainer.config["single_spec"]:
        result = trainer._map_reduce(result, num_sample)
    trainer.wandblogger.log_eval_metrics(result, head="eval")
    return result