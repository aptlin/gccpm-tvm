import os

import cv2
import numpy as np
import requests
import torch

from models.single_person_pose_with_mobilenet import (
    SinglePersonPoseEstimationWithMobileNet,
)
from modules.constants import NORMALIZATION
from modules.load_state import load_state

BASE_DIR = "./scratchpad/weights"
WEIGHTS = {
    "CocoSingle": (
        "https://www.dropbox.com/s/m13237lskmvgpdg/coco_checkpoint.pth?raw=1",
        {"num_heatmaps": 17, "num_refinement_stages": 5},
    ),
    "Lip": (
        "https://www.dropbox.com/s/yrt12s9qug9wpxz/lip_checkpoint.pth?raw=1",
        {"num_heatmaps": 16, "num_refinement_stages": 5},
    ),
}


def download(model_name, base_dir=BASE_DIR):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
    response = requests.get(WEIGHTS[model_name][0])
    with open(os.path.join(base_dir, model_name), "wb") as f:
        f.write(response.content)


def load_single_pose_model(model_name, **kwargs):
    base_dir = kwargs.pop("base_dir", None)
    if not base_dir:
        base_dir = BASE_DIR

    device = kwargs.pop("device", "cpu")

    download(model_name, base_dir)

    model = SinglePersonPoseEstimationWithMobileNet(**WEIGHTS[model_name][1], **kwargs)
    checkpoint = torch.load(
        os.path.join(base_dir, model_name), map_location=torch.device(device)
    )
    load_state(model, checkpoint)

    return model


def preprocess_image(image, model_name, scale_factor=1.0, resize=None):
    height, width = image.shape[0:2]
    if resize:
        height, width = resize

    target_width = int(width * scale_factor)
    target_height = int(height * scale_factor)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    if "Coco" in model_name:
        mean, std = NORMALIZATION["coco"]
        image = image / 255.0
        image = (image - mean) / std

