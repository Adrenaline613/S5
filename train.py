import torch.optim as optim
import argparse
import json

from model.net import S3Net
from model.set_seed import set_seed
from model import trainer
from model import loss
from model.dataloder import *


def main(args):

    # Load configuration
    with open(args.config) as config_file:
        config = json.load(config_file)

    # Set see and ensure reproducibility
    seed = config['seed']
    set_seed(seed)
    print(f"\nSet seed: {seed}")

    # Split dataset to get train and validation paths of each fold
    data_paths_all_folds = split_dataset(**config['data']['params'])
    print(f"Using {config['data']['name']} for train and val\n")

    for i in range(len(data_paths_all_folds)):
        fold = i + 1
        print(f"\033[32mFold {fold} of {len(data_paths_all_folds)}\033[0m")

        # Get one fold of data paths, and create datasets and dataloaders
        data_paths = data_paths_all_folds[i]
        train_paths, val_paths = data_paths[0], data_paths[1]
        train_dataset, val_dataset = make_dataset(train_paths, val_paths, training_mode=config['training_mode'])
        train_loader, val_loader = make_dataloader(train_dataset, val_dataset, config['data']['batch_size'])

        # Get model
        model = S3Net(training_mode=config['training_mode'], **config['model']['params'])
        if config['training_mode'] in ['fullyfinetune', 'freezefinetune']:
            print(f"Load pretrained model from {config['model']['ckpt_path']}")
            ckpt = torch.load(config['model']['ckpt_path'])
            model.encoder.load_state_dict(ckpt['encoder_state_dict'])

            if config['training_mode'] == 'freezefinetune':
                for param in model.encoder.parameters():
                    param.requires_grad = False
                print(f"Freeze encoder parameters")
        print(f"\033[32mModel loaded, {config['model']['name']}\033[0m")

        # Get loss function
        loss_fn = getattr(loss, config['loss']['name'])(**config['loss']['params'])
        print(f"\033[32mLoss function loaded, {config['loss']['name']}\033[0m")

        # Get optimizer
        optimizer = getattr(optim, config['optimizer']['name'])(model.parameters(), **config['optimizer']['params'])
        print(f"\033[32mOptimizer loaded, {config['optimizer']['name']}\033[0m")

        # Get name
        name = '_'.join([config['data']['name'], config['model']['name'], config["training_mode"], str(config["seed"])])

        # Initialize trainer
        _trainer = getattr(trainer, config['trainer']['name'])(
            name=name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=loss_fn,
            optimizer=optimizer,
            cross_val_idx=fold,
            **config['trainer']['params']
        )
        print(f"\033[32m{config['trainer']['name']} loaded, Start training\033[0m")

        # Train the model
        _trainer.fit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train S3Net')
    # parser.add_argument('--config', type=str, default='config/mass_cl_config.json', help='Path to the config file')
    parser.add_argument('--config', type=str, default='config/moda_sp_config.json', help='Path to the config file')
    args = parser.parse_args()

    main(args)
