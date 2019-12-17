'''
We are going to learn a latent space and a generative model for the MNIST dataset.

'''

import math
import numpy as np
import torch
import torch.utils
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
from torch.autograd import Variable
from VAE_with_normalizing_flows.VAE_with_flows import VariationalAutoencoderWithFlows
from VAE_with_normalizing_flows.utils.code_to_load_the_dataset import load_MNIST_dataset
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

def show_images(images, title=None, path=None):
    images = utils.make_grid(images)
    show_image(images[0], title, path)

def show_image(img, title = "", path = None):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    if path is not None:
        plt.savefig(path)
    plt.show()

# We use this custom binary cross entropy
# def binary_cross_entropy(r, x):
#     return -torch.sum(x * torch.log(r + 1e-8) + (1 - x) * torch.log(1 - r + 1e-8), dim=-1)

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()
ORIGINAL_BINARIZED_MNIST = True
use_cuda = torch.cuda.is_available()
print('Do we get access to a CUDA? - ', use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")
BATCH_SIZE = 100
HIDDEN_LAYERS = [400]
Z_DIM = 40
N_FLOWS = 10

N_EPOCHS = 200
LEARNING_RATE = 1e-5
MOMENTUM = 0.9
WEIGHT_DECAY = -1

AMORTIZED_WEIGHTS = True

N_SAMPLE = 64

SAVE_MODEL_EPOCH = N_EPOCHS - 5
PATH = 'saved_models/'

## we have the binarized MNIST
## TRAIN SET
if ORIGINAL_BINARIZED_MNIST:
    train_loader, val_loader, test_loader = load_MNIST_dataset('Original_MNIST_binarized/', BATCH_SIZE, True, True, True)
else:
    training_set = datasets.MNIST('../MNIST_dataset', train=True, download=True,
                       transform=transforms.ToTensor())
    print('Number of examples in the training set:', len(training_set))
    print('Size of the image:', training_set[0][0].shape)
    ## we plot an example only to check it
    idx_ex = 1000
    x, y = training_set[idx_ex] # x is now a torch.Tensor
    plt.imshow(x.numpy()[0], cmap='gray')
    plt.title('Example n {}, label: {}'.format(idx_ex, y))
    plt.show()

    ### we only check if it is binarized
    input_dim = x.numpy().size
    print('Size of the image:', input_dim)

    flatten_bernoulli = lambda x: transforms.ToTensor()(x).view(-1).bernoulli()

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../MNIST_dataset', train=True, transform=flatten_bernoulli),
        batch_size=BATCH_SIZE, shuffle=True)

    ## TEST SET
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../MNIST_dataset', train=False, transform=flatten_bernoulli),
    batch_size=BATCH_SIZE, shuffle=True)

    ## another way to plot some images from the dataset
    dataiter = iter(train_loader)
    images, labels = dataiter.next() ## next return a complete batch --> BATCH_SIZE images
    show_images(images.view(BATCH_SIZE,1,28,28))


## now we have our train and test set
## we can create our model and try to train it
model = VariationalAutoencoderWithFlows(28*28, HIDDEN_LAYERS, Z_DIM, N_FLOWS, amortized_params_flow = AMORTIZED_WEIGHTS)
print('Model overview and recap\n')
print(model)
print('\n')

optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE, momentum = MOMENTUM)

## training loop
training_loss = []
approx_kl = []
anal_kl = []
print('.....Starting trianing')
t = 0
for epoch in range(N_EPOCHS):
    tmp_elbo = 0
    tmp_kl = 0
    tmp_recon = 0
    n_batch = 0
    for i, data in enumerate(train_loader, 0):
        beta = min(1, 0.01 + t / 700)
        n_batch += 1
        if ORIGINAL_BINARIZED_MNIST:
            images = data
        else:
            images, labels = data
        images = images.to(device)

        reconstruction = model(images)
        # print('images shape', images.shape)
        # print('recon shape', test_set_reconstruction.shape)

        # likelihood = -binary_cross_entropy(reconstruction, images)
        likelihood = - F.binary_cross_entropy(reconstruction, images, reduction='sum')
        kl = torch.sum(model.qz - beta * model.pz)
        bound = beta * torch.sum(likelihood) - kl

        L = - bound / len(images) #BATCH_SIZE

        L.backward()
        optimizer.step()
        optimizer.zero_grad()
        # if L.item()/len(images) > 4:
        #     print('Epoch: {}, Batch: {}, images in the batch: {}, L.item: {}'.format(epoch, i, len(images), L.item()))
        training_loss.append(-bound/ len(images))
        tmp_elbo += - L.item() * BATCH_SIZE
        tmp_recon += torch.sum(likelihood)
        tmp_kl += kl

        ## we should update our t
        t += 1


    ## at the end of each epoch we can try to store some images
    ##
    with torch.no_grad():
        for r, data in enumerate(test_loader, 0):
            if ORIGINAL_BINARIZED_MNIST:
                images = data
            else:
                images, labels = data
            images = images.to(device)
            reconstruction = model(images)
            # print(test_set_reconstruction.shape)
            recon_image_ = reconstruction.view(reconstruction.shape[0], 1, 28, 28)
            images = images.view(images.shape[0], 1, 28, 28)
            if r % 100 == 0:
                # show_images(images, 'original')
                # show_images(recon_image_, 'test_set_reconstruction')
                grid1 = torchvision.utils.make_grid(images)
                writer.add_image('orig images', grid1, 0)
                grid2 = torchvision.utils.make_grid(recon_image_)
                writer.add_image('recon images', grid2)
                writer.close()
                ## maybe we just store the test_set_reconstruction
                ## maybe we just store the test_set_reconstruction
                images = utils.make_grid(images)
                recon_image_ = utils.make_grid(recon_image_)
                plt.imshow(images[0], cmap='gray')
                plt.title('Original from epoch {}'.format(epoch + 1))
                plt.savefig('reconstruction_during_training/originals_epoch_{}_example_{}'.format(epoch + 1, r))
                plt.imshow(recon_image_[0], cmap='gray')
                plt.title('Reconstruction from epoch {}'.format(epoch + 1))
                plt.savefig('reconstruction_during_training/reconstruction_epoch_{}_example_{}'.format(epoch + 1, r))

        model.eval()
        ## we want also to sample something from the model during training
        rendom_samples = model.sample(N_SAMPLE)
        samples = rendom_samples.view(rendom_samples.shape[0], 1, 28, 28)
        samples = utils.make_grid(samples)
        plt.imshow(samples[0], cmap='gray')
        plt.title('Samples from epoch {}'.format(epoch + 1))
        plt.savefig('samples_during_training/samples_epoch_{}'.format(epoch + 1))



    print('Epoch: {}, Elbo: {}, recon_error: {}, kl: {}'.format(epoch+1, tmp_elbo/ len(train_loader.dataset), -tmp_recon/ len(train_loader.dataset), tmp_kl/ len(train_loader.dataset)))

    if epoch + 1 > SAVE_MODEL_EPOCH:
        ## we have to store the model
        torch.save(model.state_dict(), PATH + 'VAE_flows_zdim_{}_epoch_{}_elbo_{}_learnrate_{}'.format(Z_DIM, epoch+1, tmp_elbo/ len(train_loader.dataset), LEARNING_RATE))




print('....Training ended')
fig = plt.figure()
plt.plot(training_loss, label='Bound mean per batch')
plt.legend()
plt.show()

# plt.plot(approx_kl, label='Approximated KL (mean)')
# plt.legend()
# plt.show()


model.eval()
with torch.no_grad():
    for i, data in enumerate(test_loader, 0):
        if ORIGINAL_BINARIZED_MNIST:
            images = data
        else:
            images, labels = data
        images = images.to(device)
        reconstruction = model(images)
        # print(test_set_reconstruction.shape)
        recon_image_ = reconstruction.view(reconstruction.shape[0], 1, 28, 28)
        images = images.view(images.shape[0], 1, 28, 28)
        if i % 100 == 0:
            show_images(images, 'original')
            show_images(recon_image_, 'test_set_reconstruction')
            images = utils.make_grid(images)
            recon_image_ = utils.make_grid(recon_image_)
            plt.imshow(images[0], cmap='gray')
            plt.title('Original')
            plt.savefig('reconstruction_during_training/originals_example_{}'.format(i))
            plt.imshow(recon_image_[0], cmap='gray')
            plt.title('Reconstruction')
            plt.savefig('reconstruction_during_training/reconstruction_example_{}'.format(i))


    # samples form the prios
    for i in range(5):
        # random_latent = torch.randn((N_SAMPLE, Z_DIM), dtype = torch.float).to(device)
        images_from_random = model.sample(N_SAMPLE)
        sampled_ima = images_from_random.view(images_from_random.shape[0], 1, 28, 28)
        show_images(sampled_ima, 'Random sampled imagess', 'random_samples/Random_samples_ex_{}'.format(i+1))
