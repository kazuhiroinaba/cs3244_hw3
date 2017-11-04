import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms, utils
import matplotlib.pyplot as plt
import time
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

plt.ion()   # interactive mode

# %%
# X_train = np.load('X_train.npy')
# X_test = np.load('X_test.npy')
# full_train = np.concatenate((X_train, X_test), axis=0)
# full_mean = np.mean(full_train)
# full_std = np.std(full_train)
full_mean = 132.35524
full_std = 44.076332


class FacesDataset(Dataset):
    """Faces dataset."""

    def __init__(self, X, y, transform=None):
        self.image_vectors = X
        self.targets = y
        self.transform = transform

    def __len__(self):
        return len(self.image_vectors)

    def __getitem__(self, idx):
        one_dim_image = self.image_vectors[idx]
        image = np.reshape(one_dim_image, (-1, 37))

        if self.transform:
            image = self.transform(image)

        return (image, target)


# %% test the dataset class
face_dataset = FacesDataset('X_test.npy')
fig = plt.figure()

for i in range(len(face_dataset)):
    sample = face_dataset[i]
    print('image number:', i)
    ax = plt.subplot(1, 4, i+1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    plt.imshow(sample(0))

    if i == 3:
        plt.show()
        break

# todo: add more transforms later
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[full_mean], std=[full_std])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[full_mean], std=[full_std])
    ]),
}

# %% loading the data for real
X_train_raw = np.load('X_train.npy')
y_train_raw = np.load('y_train.npy')

X_train, X_val, y_train, y_val = train_test_split(X_train_raw,
                                                  y_train_raw,
                                                  random_state=1)

faces_datasets = {
    'train': FacesDataset(X_train, y_train,
                          transform=data_transforms['train']),
    'val': FacesDataset(X_val, y_val, transform=data_transforms['test'])
}

dataloaders = {x: DataLoader(faces_datasets[x], batch_size=4,
                             shuffle=True, num_workers=4)
               for x in ['train', 'val']}

dataset_sizes = {x: len(faces_datasets[x]) for x in ['train', 'val']}

use_gpu = torch.cuda.is_available()
# %% model code


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0

            for data in dataloaders[phase]:
                inputs, labels = data

                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()
    time_elasped = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elasped // 60, time_elasped % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model


    
