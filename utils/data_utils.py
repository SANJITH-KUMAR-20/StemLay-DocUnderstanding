from PIL import Image
import os
import torch
import cv2 as cv
from torchvision.transforms import Compose, transforms

def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]

def load_image(image_path):
    image = cv.imread(image_path)
    w, h = image.shape[0], image.shape[1]
    image = torch.tensor(image)
    print(image.shape)
    transform = Compose([transforms.Resize((224,224))])
    img = transform(image.permute(2,0,1))
    print(img.shape)
    return img , (w,h)
