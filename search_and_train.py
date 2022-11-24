from train_utils import train, validate, test
from model import CustomNet
from datasets import get_datasets, get_data_loaders
from config import (
    MAX_NUM_EPOCHS, GRACE_PERIOD, EPOCHS, CPU, GPU,
    NUM_SAMPLES, DATA_ROOT_DIR, NUM_WORKERS
)

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import os


def train_and_validate(config):
    # Get all the datasets
    (
        train_dataset, valid_dataset, test_dataset, class_names
    ) = get_datasets(DATA_ROOT_DIR)
    print(f"[INFO]: Number of training samples: {len(train_dataset)}")
    print(f"[INFO]: Number of validation samples: {len(valid_dataset)}")
    print(f"[INFO]: Number of test samples: {len(test_dataset)}")
    # Get training and validation data loaders,
    # ignore test data loader for now.
    train_loader, valid_loader, _ = get_data_loaders(
        train_dataset, valid_dataset, test_dataset,
        config['batch_size'], NUM_WORKERS
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Initialize the model
    model = CustomNet(
        config['first_conv_out'], config['first_fc_out']
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=config['lr'], momentum=0.9
    )

    # start the training
    for epoch in range(EPOCHS):
        print(f"[INFO]: Epoch {epoch+1} of {EPOCHS}")
        train_epoch_loss, train_epoch_acc = train(
            model, train_loader, optimizer, criterion, device
        )
        valid_epoch_loss, valid_epoch_acc = validate(
            model, valid_loader, criterion, device
        )
  
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        print('-'*50)

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, 'checkpoint')
            torch.save((model.state_dict(), optimizer.state_dict()), path)
        tune.report(
            loss=valid_epoch_loss, accuracy=valid_epoch_acc
        )

def run_search():
    # Define the parameter search configuration.
    config = {
        "first_conv_out": 
            tune.sample_from(lambda _: 2 ** np.random.randint(4, 8)),
        "first_fc_out": 
            tune.sample_from(lambda _: 2 ** np.random.randint(4, 8)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16])
    }

    # Schduler to stop bad performing trails.
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=MAX_NUM_EPOCHS,
        grace_period=GRACE_PERIOD,
        reduction_factor=2)

    # Reporter to show on command line/output window
    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "training_iteration"])
    # Start run/search
    result = tune.run(
        train_and_validate,
        resources_per_trial={"cpu": CPU, "gpu": GPU},
        config=config,
        num_samples=NUM_SAMPLES,
        scheduler=scheduler,
        local_dir='raytune_result',
        keep_checkpoints_num=1,
        checkpoint_score_attr='min-validation_loss',
        progress_reporter=reporter
    )

    # Extract the best trial run from the search.
    best_trial = result.get_best_trial(
        'loss', 'min', 'last'
    )
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation acc: {best_trial.last_result['accuracy']}")

    # Carry out the final testing with the best settings.
    device = 'cuda:0'
    train_dataset, valid_dataset, test_dataset, _ = get_datasets(
        DATA_ROOT_DIR
    )
    _, _, test_loader = get_data_loaders(
        train_dataset, valid_dataset, test_dataset, 
        best_trial.config['batch_size'], NUM_WORKERS
    )
    print('[INFO]: Building best model for testing...')
    best_model = CustomNet(
        best_trial.config['first_conv_out'], 
        best_trial.config['first_fc_out']
    ).to(device)
    best_checkpoint_dir = best_trial.checkpoint.value
    print('[INFO]: Loading best model weights...')
    model_state, optimizer_state = torch.load(
        os.path.join(best_checkpoint_dir, 'checkpoint')
    )
    best_model.load_state_dict(model_state)
    test_acc = test(best_model, test_loader, device)
    print(f"[INFO]: Test results from the best trained model: {test_acc:.3f}")

if __name__ == '__main__':
    run_search()
