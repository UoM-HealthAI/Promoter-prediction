# import warnings
# from pydantic._internal._generate_schema import UnsupportedFieldAttributeWarning
# warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import FluorToBinaryCausal
from dataset import FluorescenceDataset
from tqdm import tqdm
from early_stopping import EarlyStopping
from torch import nn
import wandb
from datetime import datetime
import time
import os
import numpy as np

def train():
    # SETTINGS
    project_name = 'Promotor_States_Prediction'
    num_epochs = 1000
    lr = 1e-5
    train_batch_size = 1024
    test_batch_size = 1024
    save_interval = 50
    patience = 30
    num_workers = 8
    log_path = '../logs'
    train_data_path = '../dataset/train/'
    test_data_path = '../dataset/test/'

    # Set the environment variable
    os.environ["WANDB_API_KEY"] = "65e89d6040ee39f44b12f957c13c2af040aed83e"

    # Set the device
    device_ids = [0, 1]  # 使用多GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if device.type == "mps":
        print("Using Apple Silicon GPU")
        device_ids = [0]

    # Set the dataset
    train_dataset = FluorescenceDataset(data_path=train_data_path)
    test_dataset = FluorescenceDataset(data_path=test_data_path)
    train_loader  = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True,
                              prefetch_factor=2, persistent_workers=True
                              )
    test_loader  = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True, drop_last=True,
                            prefetch_factor=2, persistent_workers=True
                            )

    # Set the models
    model = FluorToBinaryCausal(d_model=256, nhead=8, num_layers=8, ff=512, dropout=0.1, max_len=200)
    model = nn.DataParallel(model, device_ids=device_ids)  # 多GPU
    model.to(device)

    # Set the loss function
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).to(device))

    # Set the early stopping
    early_stop = EarlyStopping(log_path=log_path, patience=patience, verbose=True)

    # Set the optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Set the wandb configuration
    current_step = 1
    wandb.init(
        # set the wandb project where this run will be logged
        project = project_name,
        # track hyperparameters and run metadata
        config = {
            "epochs": num_epochs,
            "batches_train": train_batch_size,
            "batches_test": test_batch_size,
            "learning_rate": lr,
            "optimizer": "AdamW",
            "loss_function": "BCEWithLogitsLoss",
            "patience": patience,
        },
        # Current date and time is set as the name of this run
        name = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )

    # ------- Print information
    print("--------------------------------------------------------------")
    print("The device is: ", device)
    print("The number of training data is: ", len(train_dataset))
    print("The number of test data is: ", len(test_dataset))
    print("The number of parameters in the models: ", sum(p.numel() for p in model.parameters()))
    print("The number of trainable parameters in the models: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("The batch size of training data is: ", train_batch_size)
    print("The batch size of test data is: ", test_batch_size)
    print("The learning rate is: ", lr)
    print("The number of epochs is: ", num_epochs)
    print("--------------------------------------------------------------")

    # Color definition
    BLUE_COLOR  = "\033[34m"
    RED_COLOR   = "\033[31m"
    RESET_COLOR = "\033[0m"

    start_time = time.time()
    train_epoch_loss = []
    test_epoch_loss = []

    # Train and test the models
    for epoch in range(num_epochs):

        # ================================================================================
        #                                   Training
        # ================================================================================
        # Set the total loss
        total_loss = 0.0

        # Set the progress bar
        pbar = tqdm(train_loader, total=len(train_loader), ncols=160, colour="red")

        # Set the models to train mode
        model.train()
        for i, (data, target) in enumerate(pbar):
            # Move the data to the device
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            output = model(data)
            # Compute the loss
            loss = criterion(output, target)
            # Backward pass
            loss.backward()
            # Update the parameters
            optimizer.step()

            # Add the mini-batch training loss to epoch loss
            total_loss += loss.item()

            # ------- Update the progress bar
            tqdm_epoch = f"{'Epoch:':<6}{f'{epoch+1:03d}/{num_epochs}':>8}{' | Train ':<8}"
            tqdm_lr = f"{'| LR:':<5}{f'{lr:12.8f}':>15}{' '}"
            tqdm_loss = f"{'| Loss:':<5}{f'{total_loss / (i + 1):12.8f}':>15}{' '}"
            tqdm_time = f"{'| Time:':<5}{f'{(time.time() - start_time) / 3600.0:8.4f}':>10}{' hours'}"
            s = f"{RED_COLOR}{tqdm_epoch + tqdm_lr + tqdm_loss + tqdm_time}{RESET_COLOR}"
            pbar.set_description(s)

        # Log the loss
        train_epoch_loss.append(total_loss / len(train_loader))
        wandb.log({"train_loss": total_loss / len(train_loader)}, step=current_step)

        # ================================================================================
        #                                   Testing
        # ================================================================================

        # Set the total loss
        total_loss = 0.0

        # Set the progress bar
        pbar = tqdm(test_loader, total=len(test_loader), ncols=160, colour="yellow")

        # Set the models to evaluation
        model.eval()
        with torch.no_grad():
            for i, (data, target) in enumerate(pbar):
                # Move the data to the device
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                # Forward pass
                output = model(data)
                # Compute the loss
                loss = criterion(output, target)

                # Add the mini-batch training loss to epoch loss
                total_loss += loss.item()

                # # ------- Update the progress bar
                tqdm_epoch = f"{'':<14}{'   Valid ':<8}"
                tqdm_lr = f"{'| LR:':<5}{f'{lr:12.8f}':>15}{' '}"
                tqdm_loss = f"{'| Loss:':<5}{f'{total_loss / (i + 1):12.8f}':>15}{' '}"
                tqdm_time = f"{'| Time:':<5}{f'{(time.time() - start_time) / 3600.0:8.4f}':>10}{' hours'}"
                s = f"{RED_COLOR}{tqdm_epoch + tqdm_lr + tqdm_loss + tqdm_time}{RESET_COLOR}"
                pbar.set_description(s)

        # Log the loss
        test_epoch_loss.append(total_loss / len(test_loader))
        wandb.log({"test_loss": total_loss / len(test_loader)}, step=current_step)
        current_step += 1

        # NOTE: Save models parameters when test loss decreases. If test
        #  loss doesn't decrease after a given patience, early stops the training.
        early_stop(total_loss / len(test_loader), model)
        if early_stop.early_stop:
            print(f"\n{RED_COLOR}Early stopping happened at No.{epoch+1} epoch.\n{RESET_COLOR}")
            break

        # Save models parameters at a given interval
        if (epoch+1) % save_interval == 0:
            print(f"{BLUE_COLOR}Saving parameters at No.{epoch + 1} epoch...{RESET_COLOR}")
            torch.save(model.state_dict(), log_path + '/epoch' + str(epoch+1) + '_param.pth')

    # Finish the wandb
    wandb.finish()

    # Save the best models
    model.load_state_dict(torch.load(early_stop.log_path + '/best_param.pth', weights_only=True))  # Load the best parameters
    torch.save(model, log_path + '/best_model.pt')

    # Save loss and additional information
    np.savetxt(log_path + '/train_loss.txt', np.array(train_epoch_loss))
    np.savetxt(log_path + '/test_loss.txt', np.array(test_epoch_loss))
    np.savetxt(log_path + '/train_time.txt', np.array([(time.time() - start_time) / 3600.0,]))


if __name__ == "__main__":
    train()
