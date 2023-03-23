import torch
from torchvision import datasets, transforms

class DataUtils:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_data(self, batch_size=32):
        train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.transforms.RandomRotation(35),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                       std=[0.229, 0.224, 0.225])])

        test_transforms = transforms.Compose([transforms.Resize(224),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                        std=[0.229, 0.224, 0.225])])
        
        val_transforms =  transforms.Compose([transforms.Resize(224),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                        std=[0.229, 0.224, 0.225])])

        train_dir = self.data_dir + '/train'
        valid_dir = self.data_dir + '/valid'
        test_dir = self.data_dir + '/test'
        train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
        test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
        val_data = datasets.ImageFolder(valid_dir, transform=val_transforms)

        trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
        validloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)

        return trainloader, validloader, testloader, train_data