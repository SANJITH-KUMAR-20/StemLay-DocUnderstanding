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