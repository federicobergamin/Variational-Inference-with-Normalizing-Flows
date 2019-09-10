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

# We use this custom BCE function until PyTorch implements reduce=False
def binary_cross_entropy(r, x):
    return -torch.sum(x * torch.log(r + 1e-8) + (1 - x) * torch.log(1 - r + 1e-8), dim=-1)

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()

use_cuda = torch.cuda.is_available()
print('Do we get access to a CUDA? - ', use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")
BATCH_SIZE = 100
HIDDEN_LAYERS = [400]
Z_DIM = 40
N_FLOWS = 10

N_EPOCHS = 150
LEARNING_RATE = 1e-5
MOMENTUM = 0.9
WEIGHT_DECAY = -1

AMORTIZED_WEIGHTS = True

N_SAMPLE = 64

SAVE_MODEL_EPOCH = N_EPOCHS + 5
PATH = 'saved_models/'

## we have the binarized MNIST
## TRAIN SET
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
model = VariationalAutoencoderWithFlows(input_dim, HIDDEN_LAYERS, Z_DIM, N_FLOWS, amortized_params_flow = AMORTIZED_WEIGHTS)
print('Model overview and recap\n')
print(model)
print('\n')

# for params in model.parameters():
#     print(params)


# ## optimization
# if WEIGHT_DECAY > 0:
#     # we add small L2 reg as in the original paper
#     optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY)
# else:
#     optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))

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
        images, labels = data
        images = images.to(device)

        reconstruction = model(images)
        # print('images shape', images.shape)
        # print('recon shape', conditional_reconstruction.shape)

        likelihood = -binary_cross_entropy(reconstruction, images)
        # print(likelihood)
        # likelihood = - F.binary_cross_entropy(reconstruction, images, reduction='sum')

        # print('likel hsape', likelihood.shape)
        # print('likelihood', torch.sum(likelihood))
        # print('kl', torch.sum(model.kl_divergence))
        # print('--')
        kl = torch.sum(model.qz - beta * model.pz)
        # elbo = likelihood - model.kl_divergence
        # bound = - torch.sum(model.qz) + beta * torch.sum(likelihood) + beta * torch.sum(model.pz)
        bound = beta * torch.sum(likelihood) - kl
        #
        # print('Sampled kl', model.kl_divergence.shape)

        # approx_kl.append(kl / len(images))


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
            images, labels = data
            images = images.to(device)
            reconstruction = model(images)
            # print(conditional_reconstruction.shape)
            recon_image_ = reconstruction.view(reconstruction.shape[0], 1, 28, 28)
            images = images.view(images.shape[0], 1, 28, 28)
            if r % 100 == 0:
                # show_images(images, 'original')
                # show_images(recon_image_, 'conditional_reconstruction')
                grid1 = torchvision.utils.make_grid(images)
                writer.add_image('orig images', grid1, 0)
                grid2 = torchvision.utils.make_grid(recon_image_)
                writer.add_image('recon images', grid2)
                writer.close()
                ## maybe we just store the conditional_reconstruction
                ## maybe we just store the conditional_reconstruction
                images = utils.make_grid(images)
                recon_image_ = utils.make_grid(recon_image_)
                plt.imshow(images[0], cmap='gray')
                plt.title('Original from epoch {}'.format(epoch + 1))
                plt.savefig('reconstruction_during_training/originals_epoch_{}_example_{}'.format(epoch + 1, r))
                plt.imshow(recon_image_[0], cmap='gray')
                plt.title('Reconstruction from epoch {}'.format(epoch + 1))
                plt.savefig('reconstruction_during_training/reconstruction_epoch_{}_example_{}'.format(epoch + 1, r))

        ## we want also to sample something from the model during training
        rendom_samples = model.sample(N_SAMPLE)
        samples = rendom_samples.view(rendom_samples.shape[0], 1, 28, 28)
        samples = utils.make_grid(samples)
        plt.imshow(samples[0], cmap='gray')
        plt.title('Samples from epoch {}'.format(epoch + 1))
        plt.savefig('samples_during_training/samples_epoch_{}'.format(epoch + 1))



    print('Epoch: {}, Elbo: {}, recon_error: {}, kl: {}'.format(epoch+1, tmp_elbo/ len(train_loader.dataset), tmp_recon/ len(train_loader.dataset), tmp_kl/ len(train_loader.dataset)))

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
        images, labels = data
        images = images.to(device)
        reconstruction = model(images)
        # print(conditional_reconstruction.shape)
        recon_image_ = reconstruction.view(reconstruction.shape[0], 1, 28, 28)
        images = images.view(images.shape[0], 1, 28, 28)
        if i % 100 == 0:
            show_images(images, 'original')
            show_images(recon_image_, 'conditional_reconstruction')
            images = utils.make_grid(images)
            recon_image_ = utils.make_grid(recon_image_)
            plt.imshow(images[0], cmap='gray')
            plt.title('Original')
            plt.savefig('reconstruction_during_training/originals_example_{}'.format(i))
            plt.imshow(recon_image_[0], cmap='gray')
            plt.title('Reconstruction')
            plt.savefig('reconstruction_during_training/reconstruction_example_{}'.format(i))

## at this point I want to take the test set and compute the latent code
## for each example and then run PCA or TSNE and plot it
# latent_representation = []
# all_labels = []
# with torch.no_grad():
#     for i, data in enumerate(test_loader, 0):
#         images, labels = data
#         labels = labels.numpy()
#         images = images.to(device)
#         for k in range(len(images)):
#             latent_repr0, _, _ = model.encoder(images[k])
#             latent_repr = model.flows(latent_repr0)
#             latent_representation.append(latent_repr.numpy())
#             all_labels.append(labels[k])
#
#     # at this point the two sets contain what we want
#     # we can do PCA and plot the 2 components results
#     latent_representation = np.array(latent_representation)
#     print(latent_representation.shape)
#     pca = PCA(2)
#     pca.fit(latent_representation)
#     feat = pca.fit_transform(latent_representation)
#     features_pca = np.array(feat)
#     print(features_pca.shape)
#
#     colors = ['#0165fc', '#02ab2e', '#fdaa48', '#fffe7a', '#6a79f7', '#db4bda', '#0ffef9', '#bd6c48', '#fea993', '#1e9167']
#
#     COLORS = ["#0072BD",
#               "#D95319",
#               "#006450",
#               "#7E2F8E",
#               "#77AC30",
#               "#EDB120",
#               "#4DBEEE",
#               "#A2142F",
#               "#191970",
#               "#A0522D"]
#
#     # print(all_labels)
#     all_labels = np.array(all_labels)
#     fig = plt.figure()
#     for i in range(10):
#         idxs = np.where(all_labels == i)
#         # print(idxs)
#         plt.scatter(features_pca[idxs,0], features_pca[idxs,1], c = colors[i], label = i)
#
#     # plt.scatter(features_pca[:,0], features_pca[:,1], c = all_labels)
#     plt.title('PCA on the latent dimension')
#     plt.legend()
#     plt.savefig('PCA/PCA_latent_repr_layer_nlatent_{}'.format(Z_DIM))
#     plt.show()


    ## now we want also to try to sample from the decoder
    ## RANDOM SAMLING
    # Z IS RANDOM N(0,1)
    # mus = torch.zeros((BATCH_SIZE,Z_DIM))
    # stds = torch.zeros((BATCH_SIZE, Z_DIM))
    # eps = torch.randn((BATCH_SIZE, Z_DIsM))
    # random_z = mus.addcmul(stds, eps)
    for i in range(5):
        # random_latent = torch.randn((N_SAMPLE, Z_DIM), dtype = torch.float).to(device)
        images_from_random = model.sample(N_SAMPLE)
        sampled_ima = images_from_random.view(images_from_random.shape[0], 1, 28, 28)
        show_images(sampled_ima, 'Random sampled imagess', 'random_samples/Random_samples_ex_{}'.format(i+1))
