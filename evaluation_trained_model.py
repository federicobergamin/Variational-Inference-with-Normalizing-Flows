''' Now that we have trained a VAE model wioth normalizing flows, we want to evalueate it in some way.
    We estimate the probability of data under the model using an importance sampling technique.
    We can write the marginal likelihood of a datapoint as:
            log p_theta(x) = log E_q [p_theta(x,z) / q_phi(z|x)]
                           ~ log 1/L sum( (p_theta(x|z) * p(z)) / q_phi(z|x) )
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
from sklearn.decomposition import PCA
from VAE_with_normalizing_flows.utils.code_to_load_the_dataset import load_MNIST_dataset

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

use_cuda = torch.cuda.is_available()
print('Do we get access to a CUDA? - ', use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")

ORIGINAL_BINARIZED_MNIST = True

BATCH_SIZE = 100
HIDDEN_LAYERS = [400]
Z_DIM = 40
N_FLOWS = 10

N_EPOCHS = 200
LEARNING_RATE = 1e-5
MOMENTUM = 0.9
WEIGHT_DECAY = -1

AMORTIZED_WEIGHTS = True

ESTIMATION_SAMPLES = 100

PATH = 'saved_models/VAE_flows_zdim_40_epoch_200_elbo_-94.84511192321777_learnrate_1e-05'

## we have the binarized MNIST
## in this case we look at the test set, since we are interested in these examples that
## were not used to train the model
if ORIGINAL_BINARIZED_MNIST:
    ## we load the original dataset by Larochelle
    train_loader, val_loader, test_loader = load_MNIST_dataset('Original_MNIST_binarized/', BATCH_SIZE, True, True,
                                                               True)
else:
    # we have the binarized MNIST
    ## TRAIN SET

    flatten_bernoulli = lambda x: transforms.ToTensor()(x).view(-1).bernoulli()

    ## TEST SET
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../MNIST_dataset', train=False, transform=flatten_bernoulli),
    batch_size=BATCH_SIZE, shuffle=True)

## we can create our model and try to train it
model = VariationalAutoencoderWithFlows(28*28, HIDDEN_LAYERS, Z_DIM, N_FLOWS, amortized_params_flow = AMORTIZED_WEIGHTS)
print('Model overview and recap\n')
print(model)
print('\n')

# now we have to load the trained model dict
model.load_state_dict(torch.load(PATH))



## now for each datapoint of the test set we want to compute the marginal likelihood

marginal_log_likelihood = 0
model.eval()

with torch.no_grad():
    for i, data in enumerate(test_loader, 0):
        if ORIGINAL_BINARIZED_MNIST:
            images = data
        else:
            images, labels = data
        images = images.to(device)

        batch_log_likelihood = torch.zeros((len(images), ESTIMATION_SAMPLES))

        for j in range(ESTIMATION_SAMPLES):
            # I have to forward the images through the model, this way we get the reconstruction
            reconstruction = model(images)
            # we should get the kl
            # kl = torch.sum(model.kl_divergence)
            kl = model.qz - model.pz
            # print(kl.shape)

            likelihood = - torch.sum(F.binary_cross_entropy(reconstruction, images, reduction = 'none'), 1) ## BATCH_SIZE element
            bound = likelihood - kl

            batch_log_likelihood[:,j] = bound

        ## at the end we have this matrix of size BATCH_SIZE x ESTIMATION_SAMPLES
        # print(batch_log_likelihood)
        log_likel = math.log(1/ESTIMATION_SAMPLES) + torch.logsumexp(batch_log_likelihood, dim = 1)
        marginal_log_likelihood += torch.sum(log_likel)

print('The marginal log likelihood we get on average on a test example is:', marginal_log_likelihood / len(test_loader.dataset))



