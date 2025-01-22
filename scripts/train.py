import torch
import torch.nn as nn
from tqdm import tqdm
from data.dataloaders import get_dataloader
from model.segmentor_model import UNet, get_fasterrcnn_model
from configs.load_config import load_config
from datetime import datetime
import os
from PIL import Image
import argparse
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def write_log(log_file, message):
    with open(log_file, "a") as f:
        f.write(f"{message}\n")

def log_dir_setup(train_run_name, config):
    if not os.path.exists(f"logs"):
        os.mkdir("logs")
    os.mkdir(f"logs/{train_run_name}")
    os.mkdir(f"logs/{train_run_name}/weights")
    os.mkdir(f"logs/{train_run_name}/output_images")
    log_file = f"logs/{train_run_name}/log.txt"

    write_log(log_file, f"Training {config['run']['task']} model")

def store_unet_img(config, store_path, img):
    mean = torch.tensor([0.3016, 0.4715, 0.5940]).to(device)
    std = torch.tensor([0.1854, 0.1257, 0.0930]).to(device)
    if config["augmentation"]["normalize"]:
        unnormalized_image = img * std[:, None, None] + mean[:, None, None]
        out_img = unnormalized_image.detach().cpu().numpy().transpose(1, 2, 0)
        out_img = np.clip(255 * out_img, 0, 255).astype("uint8")
    else:
        out_img = img.detach().cpu().numpy().transpose(1, 2, 0)
        out_img = out_img.astype("uint8")
    Image.fromarray(out_img.transpose(1, 0, 2)).save(store_path)

def store_rcnn_img(config, store_path):
    pass

def train_unet(config):
    # extract config info
    epochs = config["training"]["epochs"]
    lr = config["training"]["learning_rate"]
    train_dataloader = get_dataloader(config)
    model = UNet(in_channels=3, out_channels=3).to(device)
    criterion = nn.MSELoss()
     
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_run_name = f"{config['model']['architecture']}_batch={config['training']['batch_size']}_lr={lr}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    log_file = f"logs/{train_run_name}/log.txt"

    log_dir_setup(train_run_name, config)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        for i, (inputs, targets) in enumerate(tqdm(train_dataloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                write_log(log_file, f"Epoch {epoch+1}: Batch {i}, Loss: {loss.item()}")

        print(f"Saving model at epoch {epoch+1}")
        torch.save(model.state_dict(), f"logs/{train_run_name}/weights/epoch_{epoch+1}.pth")
        
        # save image genertaed by model at the end of each epoch
        if config["model"]["architecture"].lower() == "unet":
            store_unet_img(config, f"logs/{train_run_name}/output_images/{epoch+1}.png", outputs[0])
        elif config["model"]["architecture"].lower() == "rcnn":
            store_rcnn_img(config, f"logs/{train_run_name}/output_images/{epoch+1}.png")

def train_rcnn(config):
    epochs = config["training"]["epochs"]
    lr = config["training"]["learning_rate"]
    train_dataloader = get_dataloader(config)

    model = get_fasterrcnn_model(num_classes=20).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=0.0005
    )

    train_run_name = f"{config['model']['architecture']}_batch={config['training']['batch_size']}_lr={lr}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    log_file = f"logs/{train_run_name}/log.txt"
    log_dir_setup(train_run_name, config)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        
        # Put model into training mode
        model.train()

        for i, batch_data in enumerate(tqdm(train_dataloader)):
            images_batch, boxes_batch, labels_batch, _ = batch_data  # we won't use centroid_coords here

            # Move images to device (list of Tensors)
            images_batch = list(img.to(device) for img in images_batch)
            
            targets_batch = []
            for j in range(len(images_batch)):
                # boxes_batch[j] -> shape [N, 4], labels_batch[j] -> shape [N]
                boxes = boxes_batch[j].to(device)
                labels = labels_batch[j].to(device)

                # Create a target dictionary for each image
                target_dict = {}
                
                # If there are no boxes, create empty Tensors
                if boxes.shape[0] == 0:
                    target_dict["boxes"] = torch.zeros((0, 4), dtype=torch.float32, device=device)
                    target_dict["labels"] = torch.zeros((0,), dtype=torch.int64, device=device)
                else:
                    target_dict["boxes"] = boxes.float()       # (N, 4)
                    target_dict["labels"] = labels.long()      # (N,)
                
                targets_batch.append(target_dict)

            optimizer.zero_grad()
            
            # Forward pass (the model returns a dict of losses during training)
            loss_dict = model(images_batch, targets_batch)
            
            # Sum all the losses
            losses = sum(loss for loss in loss_dict.values())

            # Backprop
            losses.backward()
            optimizer.step()

            if i % 10 == 0:
                loss_string = ", ".join(f"{k}: {v.item():.4f}" for k, v in loss_dict.items())
                write_log(log_file, f"Epoch {epoch+1}, Batch {i}, Total Loss: {losses.item():.4f}, {loss_string}")

            # Step the scheduler if you use one
            # scheduler.step()
            print(f"Saving model at epoch {epoch+1}")
            torch.save(model.state_dict(), f"logs/{train_run_name}/weights/epoch_{epoch+1}.pth")

            # You could store some sample outputs or debug images here if desired
            if config["model"]["architecture"].lower() == "rcnn":
                store_rcnn_img(config, f"logs/{train_run_name}/output_images/{epoch+1}.png")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--debug", type=bool, default=False)
    args = parser.parse_args()
    config = load_config(args.config)

    config["run"]["mode"] = "train"
    config["debug"] = args.debug

    if config["model"]["architecture"].lower() == "unet":
        train_unet(config)
    else:
        train_rcnn(config)

if __name__ == "__main__":
    main()

 

