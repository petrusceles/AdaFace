import cv2

# Import
import os
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import math
import numpy as np
import random
from torch.utils.data import DataLoader


# Padding Transform Class
class PaddingSquare(nn.Module):
    def __init__(self, fill=0, padding_mode="constant"):
        super().__init__()
        self.fill = fill
        self.padding_mode = padding_mode

    def calculate_padding(self, img):
        w, h = img.size
        max_wh = np.max([w, h])
        padding = [0, 0]
        if w > h:
            h_padding = math.ceil((w - h) / 2)
            padding = [0, h_padding]
        elif h > w:
            w_padding = math.ceil((h - w) / 2)
            padding = [w_padding, 0]
        return padding

    def forward(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return transforms.functional.pad(
            img, self.calculate_padding(img), self.fill, self.padding_mode
        )


class ListDatasetWithIndex(Dataset):
    def __init__(self, img_list, image_is_saved_with_swapped_B_and_R=False):
        super(ListDatasetWithIndex, self).__init__()

        # image_is_saved_with_swapped_B_and_R: correctly saved image should have this set to False
        # face_emore/img has images saved with B and G (of RGB) swapped.
        # Since training data loader uses PIL (results in RGB) to read image
        # and validation data loader uses cv2 (results in BGR) to read image, this swap was okay.
        # But if you want to evaluate on the training data such as face_emore/img (B and G swapped),
        # then you should set image_is_saved_with_swapped_B_and_R=True

        self.img_list = img_list
        self.transform = transforms.Compose(
            [
                PaddingSquare(),
                transforms.Resize(size=(112, 112)),
                transforms.ToTensor(),
            ]
        )
        self.image_is_saved_with_swapped_B_and_R = image_is_saved_with_swapped_B_and_R

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        if self.image_is_saved_with_swapped_B_and_R:
            with open(self.img_list[idx], "rb") as f:
                img = Image.open(f)
                img = img.convert("RGB")
            img = self.transform(img)

        else:
            # ArcFace Pytorch
            img = cv2.imread(self.img_list[idx])
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img[:, :, :3]

            img = Image.fromarray(img)
            # img = np.moveaxis(img, -1, 0)
            img = self.transform(img)
        return img, idx


class ListDataset(Dataset):
    def __init__(self, img_list, image_is_saved_with_swapped_B_and_R=False):
        super(ListDataset, self).__init__()

        # image_is_saved_with_swapped_B_and_R: correctly saved image should have this set to False
        # face_emore/img has images saved with B and G (of RGB) swapped.
        # Since training data loader uses PIL (results in RGB) to read image
        # and validation data loader uses cv2 (results in BGR) to read image, this swap was okay.
        # But if you want to evaluate on the training data such as face_emore/img (B and G swapped),
        # then you should set image_is_saved_with_swapped_B_and_R=True

        self.img_list = img_list
        self.transform = transforms.Compose(
            [
                [
                    PaddingSquare(),
                    transforms.Resize(size=(112, 112)),
                    transforms.ToTensor(),
                ]
            ]
        )

        self.image_is_saved_with_swapped_B_and_R = image_is_saved_with_swapped_B_and_R

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image_path = self.img_list[idx]
        img = cv2.imread(image_path)
        img = img[:, :, :3]

        if self.image_is_saved_with_swapped_B_and_R:
            print("check if it really should be on")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(img)
        img = self.transform(img)
        return img, idx


def prepare_imagelist_dataloader(
    img_list, batch_size, num_workers=0, image_is_saved_with_swapped_B_and_R=False
):
    # image_is_saved_with_swapped_B_and_R: correctly saved image should have this set to False
    # face_emore/img has images saved with B and G (of RGB) swapped.
    # Since training data loader uses PIL (results in RGB) to read image
    # and validation data loader uses cv2 (results in BGR) to read image, this swap was okay.
    # But if you want to evaluate on the training data such as face_emore/img (B and G swapped),
    # then you should set image_is_saved_with_swapped_B_and_R=True

    image_dataset = ListDatasetWithIndex(img_list, image_is_saved_with_swapped_B_and_R)
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    return dataloader


def prepare_dataloader(
    img_list, batch_size, num_workers=0, image_is_saved_with_swapped_B_and_R=False
):
    # image_is_saved_with_swapped_B_and_R: correctly saved image should have this set to False
    # face_emore/img has images saved with B and G (of RGB) swapped.
    # Since training data loader uses PIL (results in RGB) to read image
    # and validation data loader uses cv2 (results in BGR) to read image, this swap was okay.
    # But if you want to evaluate on the training data such as face_emore/img (B and G swapped),
    # then you should set image_is_saved_with_swapped_B_and_R=True

    image_dataset = ListDataset(
        img_list,
        image_is_saved_with_swapped_B_and_R=image_is_saved_with_swapped_B_and_R,
    )
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    return dataloader
