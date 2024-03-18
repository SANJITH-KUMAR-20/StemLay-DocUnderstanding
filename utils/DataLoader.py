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
import logging
import random
import numpy as np
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

  def __init__(self, label_dir, image_dir, is_transforms, is_scale, max_bbox):

    self.max_bbox = max_bbox
    self.img_dir = image_dir
    self.label_dir = label_dir
    self.imgs = sorted(os.listdir(self.img_dir))
    self.labels = sorted(os.listdir(self.label_dir))
    self.classes = ["text_block", "image"]
    self.is_transforms = is_transforms
    self.is_scale = is_scale
    self.transform = Compose([
        transforms.Resize((224,224)),
        transforms.PILToTensor()
    ])

  def __len__(self):
    return len(self.imgs)

  def __getitem__(self, idx):
    img_dir = os.path.join(self.img_dir, self.imgs[idx])
    label = os.path.join(self.label_dir, self.labels[idx])
    img = Image.open(img_dir)
    W ,H = img.size
    file = json.load(open(label, "r"))
    if self.is_transforms:
      img = self.transform(img)

    bboxes = self._load_bboxes(file, W, H)
    labels = self._load_block_label(file)

    return img, bboxes, labels, img_dir

  def _scale_bboxes(Self, bbox : list, width : int, height : int) -> list:

    scaled_box = [0,0,0,0]
    width_scale = 224 / width
    height_scale = 224 / height
    scaled_box[0] = bbox[0] * width_scale
    scaled_box[1] = bbox[1]  * height_scale
    scaled_box[2] = bbox[2]  * width_scale
    scaled_box[3] = bbox[3]  * height_scale
    return scaled_box

  def _load_bboxes(self, json_File : list, width : int, height : int) -> torch.Tensor:

    result = [[0., 0., 0., 0.]]*self.max_bbox
    for i,blocks in enumerate(json_File):
      if i>=40:
        break
      block = blocks["bbox"]
      if self.is_scale:
        block = self._scale_bboxes(block, width, height)
      result[i] = block

    return torch.Tensor(result)

  def _load_block_label(Self, json_File : list) -> torch.Tensor:
    labels = []
    for i,blocks in enumerate(json_File):
      if i>=40:
        break
      if blocks["image"]:
        labels.append([0.,1.])
        continue
      labels.append([1.,0.])

    labels.extend([[0.,0.]]*abs(40-len(labels)))

    return torch.Tensor(labels)
  

class DFTR_V2(torch.utils.data.Dataset):

    def __init__(self, root_img_dir, root_label_dir):
        super(DFTR_V2, self).__init__()

        self.root_img_dir = root_img_dir
        self.root_label_dir = root_label_dir
        self.img_data = sorted(os.listdir(self.root_img_dir))
        self.label_data = sorted(os.listdir(self.root_label_dir))
        self.transform = Compose([
        transforms.Resize((256,256)),
        transforms.PILToTensor()
        ])

    def __len__(self):
        return len(self.img_data)
    
    def _get_pil_img(self, img_shape : Tuple[int]) -> PIL.Image.Image:
        try:
            H, W, C = img_shape
            img_tensor = torch.zeros((C, H, W))
            pil_transform = Compose([transforms.ToPILImage()])
            return pil_transform(img_tensor)
        except Exception as e:
            logging.error(f"error while generating PIL image {e}")
    
    def _scale_boxes(Self, bbox : list, width : int, height : int) -> list:
        try:
            scaled_box = [0,0,0,0]
            width_scale = 256 / width
            height_scale = 256 / height
            scaled_box[0] = bbox[0] * width_scale
            scaled_box[1] = bbox[1]  * height_scale
            scaled_box[2] = bbox[2]  * width_scale
            scaled_box[3] = bbox[3]  * height_scale
            return scaled_box
        except Exception as e:
            logging.error(f"error while scaling bounding boxes")
    
    def _gen_image(self, bboxes : list, img_shape : Tuple[int], H : int, W : int, is_scale : bool = True):
        pil_img = self._get_pil_img(img_shape)
        box_draw = PIL.ImageDraw.Draw(pil_img)
        for box in bboxes:
            if is_scale:
                box = self._scale_boxes(box, W, H)
            box_draw.rectangle(box, (255,255,255),outline = 1, width = 1)
        return pil_img
    
    def _generate_textchar_classes(self, label : Tuple[bool]) -> list:
        cls = [0., 0., 0., 0.]
        if all(label):
            cls[2] = 1.
        else:
            if label[0]:
                cls[0] = 1.
            else:
                cls[1] = 1.
        return cls
    
    def _img_label(self) -> list:
        return [0., 0., 0., 1.]
    
    def _gen_block_classes(self, blk_cls : bool) -> list:
        cls = [0., 0., 0.]
        if blk_cls:
            cls[0] = 1.
        else:
            cls[1] = 1.
        return cls
    
    def _interpolate_random_indices(self, entity : list,cls : list = None, 
                                    kind : str = "text_boxes", max_count : int = 40) -> Tuple[list] | list:
        indices = None
        if len(entity) >= max_count:
            indices = random.sample(range(len(entity)), max_count)
            indices = np.array(indices)
            if kind == "text_boxes":
                entity = list(np.array(entity)[indices])
                cls = list(np.array(cls)[indices])
                return entity,cls
            if kind == "blocks":
                entity = list(np.array(entity)[indices])
                cls = list(np.array(cls)[indices])
                return entity,cls
        else:
            rem = max_count - len(entity)
            if kind == "text_boxes":
                entity.extend([[0.,0.,0.,0.]]*rem)
                cls.extend([[0. , 0., 0., 1.]]*rem)
                return entity,cls
            if kind == "blocks":
                entity.extend([[0.,0.,0.,0.]]*rem)
                cls.extend([[0., 0., 1.]]*rem)
                return entity,cls


    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_img_dir, self.img_data[idx])
        label_path = os.path.join(self.root_label_dir, self.label_data[idx])
        img = PIL.Image.open(img_path)
        W, H = img.size
        img = self.transform(img)
        with open("dftr_data/labels/0_0_0.json", "r") as f:
            content = json.load(f)
        gen_image_boxes = []
        gen_classes = []
        block_boxes = []
        block_classes = []
        for block in content:
            try:
                block_boxes.append(self._scale_boxes(block["bbox"], W, H))
                block_classes.append(self._gen_block_classes(block["image"]))
                word_lines = block["lines"]
                for line in word_lines:
                    spans = line["spans"]
                    for span in spans:
                        box_count = 0
                        curr_boxes = [self._generate_textchar_classes((span["bold"], span["italic"]))]
                        gen_boxes = []
                        word_boxes = span["words"]
                        for bbox in word_boxes:
                            box_count += 1
                            boxx = bbox["bbox"]
                            gen_boxes.append(boxx)
                        curr_boxes = [curr_boxes] * box_count
                        gen_classes.extend(curr_boxes)
                        gen_image_boxes.extend(gen_boxes)
                    
            except Exception as e:
                pass

        gen_image_boxes, gen_classes = self._interpolate_random_indices(gen_image_boxes, gen_classes,max_count=50)
        block_boxes, block_classes = self._interpolate_random_indices(block_boxes, block_classes, kind = "blocks", max_count= 20)
        if gen_image_boxes:
            gen_img = self._gen_image(gen_image_boxes, (256, 256, 3), H, W)      
            gen_img = self.transform(gen_img)

        return img, block_boxes, block_classes, gen_img, gen_classes          







