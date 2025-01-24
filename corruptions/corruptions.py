from imagecorruptions import corrupt, get_corruption_names
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import os
from PIL import Image
import json
import torchvision
import torch

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class Data_with_Captions(torchvision.datasets.VisionDataset):

    def __init__(self, root, json_path, transform=None,task=None, corruption=None, severity=None,
                 save_path=None, loader=default_loader):
        self.root = root
        self.transform = transform
        self.corruption = corruption
        self.severity = severity
        self.save_path = save_path
        self.loader = loader

        with open(json_path, "r") as f:
            data = json.load(f)
        data = data["images"]

        a = []

        for i in range(len(data)):
            if data[i]["split"] == "test":
                a.append(data[i])

        a = a[:500]

        self.data_list = a


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, index: int):

        filename = self.data_list[index]["filename"]
        image_path = os.path.join(self.root, filename)

        image = self.loader(image_path)

        if self.transform:
            image = self.transform(image)
            # get numpy array from image
            image = np.array(image)
            image = corrupt(image, corruption_name=self.corruption, severity=self.severity)

        save_path = os.path.join(self.save_path, self.corruption, str(self.severity), "val2014")
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)


        image_name = filename
        if not os.path.exists(os.path.join(save_path)):
            os.makedirs(os.path.join(save_path), exist_ok=True)
        image_path = os.path.join(save_path, image_name)
        image = Image.fromarray(image)
        image.save(image_path)

        return 0

class Data_with_Captions_vqa(torchvision.datasets.VisionDataset):

    def __init__(self, root, json_path, transform=None,task=None, corruption=None, severity=None,
                 save_path=None, loader=default_loader):
        self.root = root
        self.transform = transform
        self.corruption = corruption
        self.severity = severity
        self.save_path = save_path
        self.loader = loader

        with open(json_path, "r") as f:
            data = json.load(f)
        data = data["questions"]
        self.img_coco_split = "val2014"
        data = data[:500]
        self.data_list = data

    def get_img_path(self, question):
        return os.path.join(
            self.root,
        f"COCO_{self.img_coco_split}_{question['image_id']:012d}.jpg"),  f"COCO_{self.img_coco_split}_{question['image_id']:012d}.jpg"


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, idx: int):

        question = self.data_list[idx]
        image_path, filename = self.get_img_path(question)

        image = Image.open(image_path)
        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)
            # get numpy array from image
            image = np.array(image)
            image = corrupt(image, corruption_name=self.corruption, severity=self.severity)

        save_path = os.path.join(self.save_path, self.corruption, str(self.severity), "val2014")
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)


        image_name = filename
        if not os.path.exists(os.path.join(save_path)):
            os.makedirs(os.path.join(save_path), exist_ok=True)
        image_path = os.path.join(save_path, image_name)
        image = Image.fromarray(image)
        image.save(image_path)

        return 0


class Data_with_Captions_okvqa(torchvision.datasets.VisionDataset):

    def __init__(self, root, json_path, transform=None,task=None, corruption=None, severity=None,
                 save_path=None, loader=default_loader):
        self.root = root
        self.transform = transform
        self.corruption = corruption
        self.severity = severity
        self.save_path = save_path
        self.loader = loader

        with open(json_path, "r") as f:
            data = json.load(f)
        data = data["questions"]
        self.img_coco_split = "val2014"
        data = data[:500]
        self.data_list = data

    def get_img_path(self, question):
        return os.path.join(
            self.root,
        f"COCO_{self.img_coco_split}_{question['image_id']:012d}.jpg"),  f"COCO_{self.img_coco_split}_{question['image_id']:012d}.jpg"


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, idx: int):

        question = self.data_list[idx]
        image_path, filename = self.get_img_path(question)

        image = Image.open(image_path)
        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)
            # get numpy array from image
            image = np.array(image)
            image = corrupt(image, corruption_name=self.corruption, severity=self.severity)

        save_path = os.path.join(self.save_path, self.corruption, str(self.severity), "val2014")
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)


        image_name = filename
        if not os.path.exists(os.path.join(save_path)):
            os.makedirs(os.path.join(save_path), exist_ok=True)
        image_path = os.path.join(save_path, image_name)
        image = Image.fromarray(image)
        image.save(image_path)

        return 0

class Data_with_Captions_vizwiz(torchvision.datasets.VisionDataset):

    def __init__(self, root, json_path, transform=None,task=None, corruption=None, severity=None,
                 save_path=None, loader=default_loader):
        self.root = root
        self.transform = transform
        self.corruption = corruption
        self.severity = severity
        self.save_path = save_path
        self.loader = loader

        with open(json_path, "r") as f:
            data = json.load(f)
        data = data["questions"]
        data = data[:500]
        self.data_list = data

    def get_img_path(self, question):
        return  os.path.join(self.root, question["image_id"]), question["image_id"]


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, idx: int):

        question = self.data_list[idx]
        image_path, filename = self.get_img_path(question)

        image = Image.open(image_path)
        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)
            # get numpy array from image
            image = np.array(image)
            image = corrupt(image, corruption_name=self.corruption, severity=self.severity)

        save_path = os.path.join(self.save_path, self.corruption, str(self.severity), "val")
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)


        image_name = filename
        if not os.path.exists(os.path.join(save_path)):
            os.makedirs(os.path.join(save_path), exist_ok=True)
        image_path = os.path.join(save_path, image_name)
        image = Image.fromarray(image)
        image.save(image_path)

        return 0


def get_args():
    parser = argparse.ArgumentParser()



    # Task Data Parameters
    parser.add_argument("--task", default="vizwiz", type=str, choices=["coco", "vqav2", "okvqa", "vizwiz"], help="the task name")
    parser.add_argument("--data_path", default=r".\eval_benchmark\vizwiz\val", type=str, help='path of the clean images')
    parser.add_argument("--json_data_path", default=r".\eval_benchmark\vizwiz\val_questions_vqa_format.json", type=str)

    # Logging Parameters
    parser.add_argument("--save_path", default=r"vizwiz_corrupted", type=str, help='the folder name of output')


    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = get_args()

    transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((336, 336))])

    for corruption in get_corruption_names():
        tic = time.time()
        for severity in range(5):
            if args.task == "vqav2":
                dataset = Data_with_Captions_vqa(root=args.data_path, json_path=args.json_data_path, transform=transforms,
                                         task=args.task,corruption=corruption, severity=severity+1, save_path=args.save_path)
            elif args.task == "coco":
                dataset = Data_with_Captions(root=args.data_path, json_path=args.json_data_path, transform=transforms,
                                         task=args.task,corruption=corruption, severity=severity+1, save_path=args.save_path)
            elif args.task == "okvqa":
                dataset = Data_with_Captions_okvqa(root=args.data_path, json_path=args.json_data_path, transform=transforms,
                                         task=args.task,corruption=corruption, severity=severity+1, save_path=args.save_path)
            elif args.task == "vizwiz":
                dataset = Data_with_Captions_vizwiz(root=args.data_path, json_path=args.json_data_path, transform=transforms,
                                         task=args.task,corruption=corruption, severity=severity+1, save_path=args.save_path)
            else:
                raise ValueError("Task not found")
            data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)

            for i, (image) in enumerate(data_loader):
                print(i)
        print(corruption, time.time() - tic)


