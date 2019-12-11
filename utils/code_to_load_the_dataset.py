import torch
import torch.utils.data as data_utils
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets, transforms, utils


def show_images(images, title=None, path=None):
    images = utils.make_grid(images)
    show_image(images[0], title, path)

def show_image(img, title = "", path = None):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    if path is not None:
        plt.savefig(path)
    plt.show()


# print('loading the datasets...')
# train = np.loadtxt('binarized_mnist_train.amat')
# valid = np.loadtxt('binarized_mnist_valid.amat')
# test = np.loadtxt('binarized_mnist_test.amat')
# print('Datatsets loaded')
# print(train.shape)
#
# plt.imshow(train[0,:].reshape(28,28), cmap='gray')
# plt.title('Digit example from the training set')
# plt.show()
#
# # batch size
# BATCH_SIZE = 64
#
# train = train.reshape(-1,28,28)
# print(train.shape)
# valid = valid.reshape(-1,28,28)
# test = test.reshape(-1,28,28)
#
# train = torch.from_numpy(train).float()
# validation = torch.from_numpy(valid).float()
# test = torch.from_numpy(test).float()
# print(train.shape)
#
# # plt.imshow(train[0,:,:], cmap='gray')
# # plt.title('Digit example from the training set')
# # plt.show()
#
# # pytorch data loader
# # train = data_utils.TensorDataset(torch.from_numpy(train).float())
# train_loader = data_utils.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
#
# # validation = data_utils.TensorDataset(torch.from_numpy(valid).float())
# val_loader = data_utils.DataLoader(validation, batch_size=BATCH_SIZE, shuffle=False)
#
# # test = data_utils.TensorDataset(torch.from_numpy(test).float())
# test_loader = data_utils.DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)
#
#
# dataiter = iter(train_loader)
# images = dataiter.next() ## next return a complete batch --> BATCH_SIZE images
# print(images.shape)
# print(images.unsqueeze(1).shape)
# show_images(images.unsqueeze(1))
#
# i = 0
# for data in train_loader:
#     i+=1
#
# print(i)
#

def load_MNIST_dataset(dir, batch_size, flatten = False, verbose = False, show_examples = False):

    print('Loading the datasets...')
    train = np.loadtxt(dir + 'binarized_mnist_train.amat')
    valid = np.loadtxt(dir + 'binarized_mnist_valid.amat')
    test = np.loadtxt(dir + 'binarized_mnist_test.amat')
    print('Datatsets loaded\n')

    if not flatten:
        train = train.reshape(-1, 28, 28)
        valid = valid.reshape(-1, 28, 28)
        test = test.reshape(-1, 28, 28)

    train = torch.from_numpy(train).float()
    validation = torch.from_numpy(valid).float()
    test = torch.from_numpy(test).float()

    if verbose:
        print('Training set shape:', train.shape)
        print('Validation et shape:', validation.shape)
        print('Test set shape', test.shape)

    train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=False)
    val_loader = data_utils.DataLoader(validation, batch_size=batch_size, shuffle=False)
    test_loader = data_utils.DataLoader(test, batch_size=batch_size, shuffle=True)

    if show_examples:
        dataiter = iter(train_loader)
        images = dataiter.next()  ## next return a complete batch --> BATCH_SIZE images
        if flatten:
            show_images(images.view(batch_size, 1, 28,28))
        else:
            show_images(images.unsqueeze(1))

    return train_loader, val_loader, test_loader




#t, v, te = load_MNIST_dataset('../Original_MNIST_binarized/', 64, True, True, True)













