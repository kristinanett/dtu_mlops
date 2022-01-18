import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import matplotlib.pyplot as plt
from PIL import Image

class customDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.Tensor(targets)
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        if self.transform:
            x = Image.fromarray(self.data[index].astype(np.uint8).transpose(1,2,0))
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)

def getFileData(path):
    data = dict(np.load(path))
    imgs = np.array(data["images"])
    labels = np.array(data["labels"])
    return imgs, labels

def mnist():
    train_data_paths = glob.glob("../../../data/corruptmnist/train*.npz")
    test_data_path  = "../../../data/corruptmnist/test.npz"

    #Obtaining the training data
    X_train = []
    y_train = []
    for path in train_data_paths:
        imgs, labels = getFileData(path)
        X_train.append(imgs)
        y_train.append(labels)

    X_train = np.array(X_train).reshape(-1, 1, 28, 28)
    y_train = np.array(y_train).flatten()

    #Obtaining the testing data
    X_test = []
    y_test = []
    imgs, labels = getFileData(test_data_path)
    X_test.append(imgs)
    y_test.append(labels)

    X_test = np.array(X_test).reshape(-1, 1, 28, 28)
    y_test = np.array(y_test).flatten()

    #Visualizing one example image
    #print(y_train[10])
    #imgplot = plt.imshow(X_train[10], cmap="gray")

    #From np arrays to dataloaders
    #transform = transforms.Compose([transforms.ToTensor(),
    #    transforms.Normalize((0.5,), (0.5,)),])

    X_train_tensor = torch.Tensor(X_train)
    y_train_tensor = torch.Tensor(y_train)

    X_test_tensor = torch.Tensor(X_test)
    y_test_tensor = torch.Tensor(y_test)

    trainset = customDataset(X_train_tensor, y_train_tensor, transform = None)
    train_dataloader = DataLoader(trainset, batch_size=4, shuffle=True)

    testset = customDataset(X_test_tensor, y_test_tensor, transform = None)
    test_dataloader = DataLoader(testset)

    print("Data obtained")
    return train_dataloader, test_dataloader
