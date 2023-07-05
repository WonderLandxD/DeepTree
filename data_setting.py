import glob
import os
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
import torch

# root = '/pub/data/chizm/BRACS/BRACS_RoI/norm_version/train'

class BRACSDatasets_I_vs_NBAUFD(Dataset):
    def __init__(self, list):
        self.roi_list = list
        self.roi_transform = transforms.Compose([transforms.Resize([1000, 1000]),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.RandomVerticalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])
                                                 ])

    def __getitem__(self, item):
        _roi = Image.open(self.roi_list[item]).convert('RGB')
        roi = self.roi_transform(_roi)
        label_name = os.path.dirname(self.roi_list[item]).split('/')[-1]
        if label_name == '6_IC':
            label = 0
        elif label_name == '0_N' or label_name == '1_PB' or label_name == '2_UDH' or label_name == '3_FEA' or label_name == '4_ADH' or label_name == '5_DCIS':
            label = 1
        # roi_path = self.roi_list[item]
        return roi, label

    def __len__(self):
        return len(self.roi_list)


class BRACSDatasets_NBU_vs_AFD(Dataset):
    def __init__(self, list):
        self.roi_list = list
        self.roi_transform = transforms.Compose([transforms.Resize([1000, 1000]),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.RandomVerticalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])
                                                 ])

    def __getitem__(self, item):
        _roi = Image.open(self.roi_list[item]).convert('RGB')
        roi = self.roi_transform(_roi)
        label_name = os.path.dirname(self.roi_list[item]).split('/')[-1]
        if label_name == label_name == '0_N' or label_name == '1_PB' or label_name == '2_UDH':
            label = 0
        elif label_name == '4_ADH' or label_name == '3_FEA' or label_name == '5_DCIS':
            label = 1
        # roi_path = self.roi_list[item]
        return roi, label

    def __len__(self):
        return len(self.roi_list)


class BRACSDatasets_N_vs_BU(Dataset):
    def __init__(self, list):
        self.roi_list = list
        self.roi_transform = transforms.Compose([transforms.Resize([1000, 1000]),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.RandomVerticalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])
                                                 ])

    def __getitem__(self, item):
        _roi = Image.open(self.roi_list[item]).convert('RGB')
        roi = self.roi_transform(_roi)
        label_name = os.path.dirname(self.roi_list[item]).split('/')[-1]
        if label_name == '0_N':
            label = 0
        elif label_name == '1_PB' or label_name == '2_UDH':
            label = 1
        # roi_path = self.roi_list[item]
        return roi, label

    def __len__(self):
        return len(self.roi_list)


class BRACSDatasets_B_vs_U(Dataset):
    def __init__(self, list):
        self.roi_list = list
        self.roi_transform = transforms.Compose([transforms.Resize([1000, 1000]),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.RandomVerticalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])
                                                 ])

    def __getitem__(self, item):
        _roi = Image.open(self.roi_list[item]).convert('RGB')
        roi = self.roi_transform(_roi)
        label_name = os.path.dirname(self.roi_list[item]).split('/')[-1]
        if label_name == '1_PB':
            label = 0
        elif label_name == '2_UDH':
            label = 1
        # roi_path = self.roi_list[item]
        return roi, label

    def __len__(self):
        return len(self.roi_list)


class BRACSDatasets_AF_vs_D(Dataset):
    def __init__(self, list):
        self.roi_list =list
        self.roi_transform = transforms.Compose([transforms.Resize([1000, 1000]),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.RandomVerticalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])
                                                 ])

    def __getitem__(self, item):
        _roi = Image.open(self.roi_list[item]).convert('RGB')
        roi = self.roi_transform(_roi)
        label_name = os.path.dirname(self.roi_list[item]).split('/')[-1]
        if label_name == '3_FEA' or label_name == '4_ADH':
            label = 0
        elif label_name == '5_DCIS':
            label = 1
        # roi_path = self.roi_list[item]
        return roi, label

    def __len__(self):
        return len(self.roi_list)


class BRACSDatasets_A_vs_F(Dataset):
    def __init__(self, list):
        self.roi_list =list
        self.roi_transform = transforms.Compose([transforms.Resize([1000, 1000]),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.RandomVerticalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])
                                                 ])

    def __getitem__(self, item):
        _roi = Image.open(self.roi_list[item]).convert('RGB')
        roi = self.roi_transform(_roi)
        label_name = os.path.dirname(self.roi_list[item]).split('/')[-1]
        if label_name == '4_ADH':
            label = 0
        elif label_name == '3_FEA':
            label = 1
        # roi_path = self.roi_list[item]
        return roi, label

    def __len__(self):
        return len(self.roi_list)


class BRACSDatasets_Test(Dataset):
    def __init__(self, list):
        self.roi_list = list
        self.roi_transfrom = transforms.Compose([transforms.Resize([1000, 1000]),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])
                                                 ])

    def __getitem__(self, item):
        _roi = Image.open(self.roi_list[item]).convert('RGB')
        roi = self.roi_transfrom(_roi)
        label_name = os.path.dirname(self.roi_list[item]).split('/')[-1]
        if label_name == '6_IC':
            # label = torch.tensor([0, 0, 0, 0])
            label = 6
        elif label_name == '0_N':
            # label = torch.tensor([1, 0, 0, 0])
            label = 0
        elif label_name == '1_PB':
            # label = torch.tensor([1, 0, 1, 0])
            label = 1
        elif label_name == '2_UDH':
            # label = torch.tensor([1, 0, 1, 1])
            label = 2
        elif label_name == '5_DCIS':
            # label = torch.tensor([1, 1, 1, 0])
            label = 5
        elif label_name == '4_ADH':
            # label = torch.tensor([1, 1, 0, 0])
            label = 4
        elif label_name == '3_FEA':
            # label = torch.tensor([1, 1, 0, 1])
            label = 3

        # roi_path = self.roi_list[item]
        return roi, label

    def __len__(self):
        return len(self.roi_list)
