import torch
import torch.nn as nn
from tqdm import tqdm
from data.dataloaders import get_dataloader
from model.segmentor_model import UNet
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

def store_img(config, store_path, img):
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

def train(config):
    # extract config info
    epochs = config["training"]["epochs"]
    lr = config["training"]["learning_rate"]
    train_dataloader = get_dataloader(config)
    model = UNet(in_channels=3, out_channels=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_run_name = f"{config['run']['task']}_batch={config['training']['batch_size']}_lr={lr}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    log_file = f"logs/{train_run_name}/log.txt"

    if not os.path.exists(f"logs"):
        os.mkdir("logs")
    os.mkdir(f"logs/{train_run_name}")
    write_log(log_file, f"Training {config['run']['task']} model")

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

        if config["debug"]:
            # input image
            store_img(config, f"logs/{train_run_name}/input_image_{epoch+1}.png", inputs[0])

            # label image
            store_img(config, f"logs/{train_run_name}/label_image_{epoch+1}.png", targets[0])

        print(f"Saving model at epoch {epoch+1}")
        torch.save(model.state_dict(), f"logs/{train_run_name}/model_epoch_{epoch+1}.pth")
        
        # save image genertaed by model at the end of each epoch
        store_img(config, f"logs/{train_run_name}/output_image_{epoch+1}.png", outputs[0])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--debug", type=bool, default=False)
    args = parser.parse_args()
    config = load_config(args.config)
    config["run"]["mode"] = "train"
    config["debug"] = args.debug
    train(config)

if __name__ == "__main__":
    main()

 

