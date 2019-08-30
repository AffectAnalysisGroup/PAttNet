
import os
import torch
from torchvision import datasets, transforms

import params
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class FaceDataset(Dataset):

    def __init__(self, csv_file, transform=None, train=True):

        self.labels_frame = pd.read_csv(csv_file, sep="\t", header=None)
        print(self.labels_frame.ix[1, 0:12].as_matrix())
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        path = '../data/patches_color/'

        img_name = self.labels_frame.ix[idx, 0]

        image1 = Image.open(path + '/patch1/' + img_name)
        image1 = image1.resize((100, 100))
        image2 = Image.open(path + '/patch2/' + img_name)
        image2 = image2.resize((100, 100))
        image3 = Image.open(path + '/patch3/' + img_name)
        image3 = image3.resize((100, 100))
        image4 = Image.open(path + '/patch4/' + img_name)
        image4 = image4.resize((100, 100))
        image5 = Image.open(path + '/patch5/' + img_name)
        image5 = image5.resize((100, 100))
        image6 = Image.open(path + '/patch6/' + img_name)
        image6 = image6.resize((100, 100))
        image7 = Image.open(path + '/patch7/' + img_name)
        image7 = image7.resize((100, 100))
        image8 = Image.open(path + '/patch8/' + img_name)
        image8 = image8.resize((100, 100))
        image9 = Image.open(path + '/patch9/' + img_name)
        image9 = image9.resize((100, 100))

        image1 = self.transform(image1)
        image2 = self.transform(image2)
        image3 = self.transform(image3)
        image4 = self.transform(image4)
        image5 = self.transform(image5)
        image6 = self.transform(image6)
        image7 = self.transform(image7)
        image8 = self.transform(image8)
        image9 = self.transform(image9)

        label = self.labels_frame.ix[idx, 1:12].as_matrix().astype('float')

        return image1, image2, image3, image4, image5, image6, image7, image8, image9, label


dataset_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(
                                            mean=params.dataset_mean,
                                            std=params.dataset_std)])


def get_bp4d41_train1(train):
    face_dataset = FaceDataset(csv_file = '../data/train_list1.txt', train=train, transform=dataset_transform)
    face_loader = torch.utils.data.DataLoader(face_dataset, batch_size=params.batch_size, shuffle=True)
    return face_loader

def get_bp4d41_test1(train):
    face_dataset = FaceDataset(csv_file = '../data/test_list1.txt', train=train, transform=dataset_transform)
    face_loader = torch.utils.data.DataLoader(face_dataset, batch_size=params.batch_size, shuffle=False)
    return face_loader

