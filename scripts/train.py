import torch
import torch.nn as nn
from tqdm import tqdm
from data.dataloaders import get_dataloader
from model.segmentor_model import UNet, get_yolo_model
from configs.load_config import load_config
from datetime import datetime
import os
from PIL import Image
import argparse
import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

def store_box_img(config, store_path, sample_image, model):
    model.eval()
    with torch.no_grad():
        outputs = model([sample_image])  # Model expects a list of images
        sample_image_cpu = sample_image.cpu()

        # unnormalize the image
        if config["augmentation"]["normalize"]:
            sample_image_cpu = sample_image.cpu()
            mean = torch.tensor([0.3016, 0.4715, 0.5940])
            std = torch.tensor([0.1854, 0.1257, 0.0930])
            img_np = sample_image_cpu * std[:, None, None] + mean[:, None, None]
            img_np = img_np.detach().cpu().numpy().transpose(2, 1, 0)
            img_np = np.clip(255 * img_np, 0, 255).astype("uint8")
        else:
            img_np = sample_image_cpu.permute(2, 1, 0).numpy()

        predictions = outputs[0]
        boxes = predictions["boxes"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()

        # Plot the image with bounding boxes
        _, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(img_np)

        for box, score, label in zip(boxes, scores, labels):
            x_min, y_min, x_max, y_max = box
            rect = patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(
                x_min,
                y_min - 5,
                f"{label}: {score:.2f}",
                color="red",
                fontsize=12,
                bbox=dict(facecolor='yellow', alpha=0.5)
            )
        
        plt.axis("off")
        plt.savefig(store_path)
        plt.close()

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
        store_unet_img(config, f"logs/{train_run_name}/output_images/{epoch+1}.png", outputs[0])

def train_bounding_box(config):
    epochs = config["training"]["epochs"]
    lr = config["training"]["learning_rate"]

    # logger setup code
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    train_dataloader = get_dataloader(config, logger)

    model = get_yolo_model(num_classes=25).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=0.0005
    )

    train_run_name = f"{config['model']['architecture']}_batch={config['training']['batch_size']}_lr={lr}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    log_file = f"logs/{train_run_name}/log.txt"
    log_dir_setup(train_run_name, config)

    first_img, _, _, _ = next(iter(train_dataloader))
    first_img = first_img[0].to(device)
    store_box_img(config, f"logs/{train_run_name}/output_images/pretrained.png", first_img, model)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        
        # Put model into training mode
        model.train()

        for i, batch_data in enumerate(tqdm(train_dataloader)):
            images_batch, boxes_batch, labels_batch, _ = batch_data  # we won't use centroid_coords here
            images_batch = [img.to(device) for img in images_batch]

            # Move images to device (list of Tensors)           
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
                    target_dict["boxes"] = boxes.float()       
                    target_dict["labels"] = labels.long()

                # check if boxes have positive area
                boxes_valid = True
                for box in boxes:
                    if box[0] >= box[2] or box[1] >= box[3]:
                        boxes_valid = False
                
                targets_batch.append(target_dict)            

            if not boxes_valid:
                continue

            optimizer.zero_grad()
            # Forward pass (the model returns a dict of losses during training)
            # logger.info(f"images_batch image shape: {images_batch[0].shape}")
            loss_dict = model(images_batch, targets_batch)
            
            # Sum all the losses
            losses = sum(loss for loss in loss_dict.values())

            # Backprop
            losses.backward()
            optimizer.step()

            if i % 10 == 0:
                loss_string = ", ".join(f"{k}: {v.item():.4f}" for k, v in loss_dict.items())
                write_log(log_file, f"Epoch {epoch+1}, Batch {i}, Total Loss: {losses.item():.4f}, {loss_string}")

        print(f"Saving model at epoch {epoch+1}")
        torch.save(model.state_dict(), f"logs/{train_run_name}/weights/epoch_{epoch+1}.pth")

        # You could store some sample outputs or debug images here if desired
        store_box_img(config, 
                    f"logs/{train_run_name}/output_images/{epoch+1}.png", 
                    images_batch[0],
                    model)

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
        train_bounding_box(config)

if __name__ == "__main__":
    main()

 

