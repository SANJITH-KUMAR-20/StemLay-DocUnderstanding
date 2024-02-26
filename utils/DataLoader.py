import json
import torch
from torchvision.transforms import Compose, transforms
import os
import cv2 as cv
import PIL
from data_utils import normalize_bbox, load_image
from PIL import ImageDraw
from utils.data_utils import load_image
from typing import Tuple
from PIL import Image


class FUNSD(torch.utils.data.Dataset):

    def __init__(self, root_dir):
        self.img_dir_path = os.path.join(root_dir,"images")
        self.annotation_dir_path = os.path.join(root_dir, "annotations")
        self.root_dir = root_dir
        self.img_dir = os.listdir(self.img_dir_path)
        self.annot_dir = os.listdir(self.annotation_dir_path)
        # self.max_annot_len, self.no_of_data = self._check_max()

    def _check_max(self) -> Tuple[int ,int]:
        max_len = 0
        skipped = []
        for annot in os.listdir(self.root_dir):
            json_path = os.path.join(self.root_dir, annot)
            try:
                with open(json_path,"r") as f:
                    data = json.load(f)
                max_len = max(max_len, len(data["form"]))
            except:
                skipped.append(json_path)
                continue
        return max_len, len(os.listdir(self.annot_dir)) - len(skipped)
    

    def get_line_bbox(self, bboxs):
        x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
        y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]

        x0, y0, x1, y1 = min(x), min(y), max(x), max(y)

        assert x1 >= x0 and y1 >= y0
        bbox = [[x0, y0, x1, y1] for _ in range(len(bboxs))]
        return bbox
    
    
    def __len__(self):
        return len(self.img_dir)
    
    def __getitem__(self, idx):
        image_path = self.img_dir_path + "//" + f"{self.img_dir[idx]}"
        annotation = self.annotation_dir_path + "//" + f"{self.annot_dir[idx]}"
        tokens = []
        bboxes = []
        boxes = []
        ner_tags = []
        with open(annotation, "r", encoding="utf8") as f:
            data = json.load(f)
        # image_path = os.path.join(img_dir, file)
        # image_path = image_path.replace("json", "png")
        image, size = load_image(image_path)
        for item in data["form"]:
            cur_line_bboxes = []
            words, label = item["words"], item["label"]
            words = [w for w in words if w["text"].strip() != ""]
            if len(words) == 0:
                continue
            if label == "other":
                for w in words:
                    tokens.append(w["text"])
                    ner_tags.append("O")
                    boxes.append(w["box"])
                    cur_line_bboxes.append(normalize_bbox(w["box"], size))
            else:
                tokens.append(words[0]["text"])
                ner_tags.append("B-" + label.upper())
                boxes.append(w["box"])
                cur_line_bboxes.append(normalize_bbox(words[0]["box"], size))
                for w in words[1:]:
                    tokens.append(w["text"])
                    ner_tags.append("I-" + label.upper())
                    boxes.append(w["box"])
                    cur_line_bboxes.append(normalize_bbox(w["box"], size))
            # by default: --segment_level_layout 1
            # if do not want to use segment_level_layout, comment the following line
            cur_line_bboxes = self.get_line_bbox(cur_line_bboxes)
            # box = normalize_bbox(item["box"], size)
            # cur_line_bboxes = [box for _ in range(len(words))]
            bboxes.extend(cur_line_bboxes)
        return {"tokens": tokens, "bboxes": bboxes, "ner_tags": ner_tags,
                         "image": image, "image_path": image_path, "boxes" : boxes}
    

class DTFR(torch.utils.data.Dataset):

  def __init__(self, label_dir, image_dir, is_transforms, is_scale):

    self.img_dir = image_dir
    self.label_dir = label_dir
    self.imgs = sorted(os.listdir(self.img_dir))
    self.labels = sorted(os.listdir(self.label_dir))
    self.classes = ["text_block", "image"]
    self.is_transforms = is_transforms
    self.is_scale = is_scale
    self.transform = Compose([
        transforms.Resize((256,256)),
        transforms.PILToTensor()
    ])

  def __len__(self):
    return len(self.imgs)

  def __getitem__(self, idx):
    img = os.path.join(self.img_dir, self.imgs[idx])
    label = os.path.join(self.label_dir, self.labels[idx])
    img = Image.open(img)
    W ,H = img.size
    file = json.load(open(label, "r"))
    if self.is_transforms:
      img = self.transform(img)

    bboxes = self._load_bboxes(file)
    labels = self._load_block_label(file)

    return img, bboxes, labels
  
  def _scale_boxes(Self, bbox : list, width : int, height : int) -> list:

    scaled_box = [0,0,0,0]
    width_scale = 256 / width
    height_scale = 256 / height
    scaled_box[0] = bbox[0] * width_scale
    scaled_box[1] = bbox[1]  * height_scale
    scaled_box[2] = bbox[2]  * width_scale
    scaled_box[3] = bbox[3]  * height_scale
    return scaled_box

  def _load_bboxes(self, json_File : list, width : int, height : int) -> torch.Tensor:

    result = []
    for blocks in json_File:
      block = blocks["bbox"]
      if self.is_scale:
        block = self._scale_bboxes(block, width, height)
      result.append(block)

    return torch.Tensor(result)

  def _load_block_label(Self, json_File : list) -> torch.Tensor:
    labels = []
    for blocks in json_File:
      if blocks["image"]:
        labels.append([0,1])
        continue
      labels.append([1,0])

    return torch.Tensor(labels)
