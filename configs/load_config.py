import yaml

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # include specific dataset paths
    if config["run"]["task"] == "segmentation":
        if config["run"]["mode"] == "train":
            config["data_path"] = "data/segmentation_dataset/train"
        else:
            config["data_path"] = "data/segmentation_dataset/test"
    elif config["run"]["task"].lower() in ["rcnn", "yolo"]:
        if config["run"]["mode"] == "train":
            config["data_path"] = "data/rcnn_segment_dataset/images/train"
        else:
            config["data_path"] = "data/rcnn_segment_dataset/images/test"
    else:
        if config["run"]["mode"] == "train":
            config["data_path"] = "data/recognition_dataset/train"
        else:
            config["data_path"] = "data/recognition_dataset/test"
    return config   