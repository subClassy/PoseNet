import os
import torch.utils.data as data
from torchvision import transforms as T
from PIL import Image
import numpy as np
import torch

class DataSource(data.Dataset):
    def __init__(self, root, resize=256, crop_size=224, train=True):
        self.root = os.path.expanduser(root)
        self.resize = resize
        self.crop_size = crop_size
        self.train = train

        self.image_poses = []
        self.images_path = []

        self._get_data()

        # TODO: Define preprocessing
        self.resize_transform = T.Resize(resize)
        if train:
            self.transform = T.Compose([
                T.RandomCrop(crop_size),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.transform = T.Compose([
                T.CenterCrop(crop_size),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        # Load mean image
        self.mean_image_path = os.path.join(self.root, 'mean_image.npy')
        if os.path.exists(self.mean_image_path):
            self.mean_image = np.load(self.mean_image_path)
            print("Mean image loaded!")
        else:
            self.mean_image = self.generate_mean_image()

    def _get_data(self):

        if self.train:
            txt_file = self.root + 'dataset_train.txt'
        else:
            txt_file = self.root + 'dataset_test.txt'

        with open(txt_file, 'r') as f:
            next(f)  # skip the 3 header lines
            next(f)
            next(f)
            for line in f:
                fname, p0, p1, p2, p3, p4, p5, p6 = line.split()
                p0 = float(p0)
                p1 = float(p1)
                p2 = float(p2)
                p3 = float(p3)
                p4 = float(p4)
                p5 = float(p5)
                p6 = float(p6)
                self.image_poses.append((p0, p1, p2, p3, p4, p5, p6))
                self.images_path.append(self.root + fname)

    def generate_mean_image(self):
        print("Computing mean image:")

        # TODO: Compute mean image

        mean_image = np.zeros((256, 455, 3))

        # Initialize mean_image
        # Iterate over all training images
        # Resize, Compute mean, etc...
        transform = T.Resize(self.resize)

        for img_path in self.images_path:
            data = Image.open(img_path)
            data = transform(data)
            mean_image += np.array(data)
            
        mean_image = mean_image / len(self.images_path)

        # Store mean image
        np.save(self.mean_image_path, mean_image)
        print("Mean image computed!")

        return mean_image

    def __getitem__(self, index):
        """
        return the data of one image
        """
        img_path = self.images_path[index]
        img_pose = self.image_poses[index]

        data = Image.open(img_path)

        # TODO: Perform preprocessing
        data = self.resize_transform(data)
        data = np.array(data)
        data = data - self.mean_image
        data = data.astype(np.uint8)
        data = Image.fromarray(data)
        data = self.transform(data)

        # img_pose = torch.Tensor(img_pose)

        return data, img_pose

    def __len__(self):
        return len(self.images_path)