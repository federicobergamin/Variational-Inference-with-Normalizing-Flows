'''
In this file we are going to create our first implementation of a VAE, following
the Kingma and Welling [2014] paper. I cannot be sure it will be an optimized version,
but I will try to do my best.


'''
import math
import torch
import torch.utils
import torch.utils.data
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init
from VAE_with_normalizing_flows.flows_with_amortized_weights import PlanarFlow, NormalizingFlows
from VAE_with_normalizing_flows.maxout import Maxout
import numpy as np

## function to compute the standard gaussian N(x;0,I) and a gaussian parametrized by
## mean mu and variance sigma log N(x|µ,σ)
def log_standard_gaussian(x):
    """
    Evaluates the log pdf of a standard normal distribution at x. (Univariate distribution)
    :param x: point to evaluate
    :return: log N(x|0,I)
    """
    return torch.sum(-0.5 * math.log(2 * math.pi) - (x ** 2 + 1e-8) / 2, dim=-1)


def log_gaussian(x, mu, log_var):
    """
    Returns the log pdf of a normal distribution parametrised
    by mu and log_var evaluated at x. (Univariate distribution)
    :param x: point to evaluate
    :param mu: mean of distribution
    :param log_var: log variance of distribution
    :return: log N(x|µ,σ)
    """
    log_pdf = - 0.5 * math.log(2 * math.pi) - (log_var + 1e-8) / 2 - ((x - mu)**2 + 1e-8) / (2 * torch.exp(log_var))
    # print('Size log_pdf:', log_pdf.shape)
    return torch.sum(log_pdf, dim=-1)

## in a simple explanation a VAE is made up of three different parts:
## - Inference model (or encoder) q_phi(z|x)
## - A stochastic layer that sample (Reparametrization trick)
## - a generative model (or decoder) p_theta(z|x)
## given this, we want to minimize the ELBO

def reparametrization_trick(mu, log_var):
    '''
    Function that given the mean (mu) and the logarithmic variance (log_var) compute
    the latent variables using the reparametrization trick.
        z = mu + sigma * noise, where the noise is sample

    :param mu: mean of the z_variables
    :param log_var: variance of the latent variables
    :return: z = mu + sigma * noise
    '''
    # we should get the std from the log_var
    # log_std = 0.5 * log_var (use the logarithm properties)
    # std = exp(log_std)
    std = torch.exp(log_var * 0.5)

    # we have to sample the noise (we do not have to keep the gradient wrt the noise)
    eps = Variable(torch.randn_like(std), requires_grad=False)
    z = mu.addcmul(std, eps)

    return z


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim,  amortized_params_flow = False, n_flows = None):
        '''
        Probabilistic inference network given by a MLP. In case of a Gaussian MLP, we will
        have to output: log(sigma^2) and mu.

        :param input_dim: dimension of the input (scalar)
        :param hidden_dims: dimensions of the hidden layers (vector)
        :param latent_dim: dimension of the latent space
        :param amortized_params_flow: bool, if true we get the u, w ,b from the input
        :param n_flows: number of flows

        In addition to return z, _mu, _log_var, if amortized_params_flow = True it returns also
        the weights needed for the transformation
        '''

        super(Encoder, self).__init__()

        self.z_dims = latent_dim
        ## now we have to create the architecture
        neurons = [input_dim, *hidden_dims]
        ## common part of the architecture
        self.hidden_layers = nn.ModuleList([nn.Linear(neurons[i-1], neurons[i]) for i in range(1,len(neurons))])
        # self.maxout = Maxout(4)
        # dim_after_maxout = int(hidden_dims[-1] / 4)
        ## we have two output: mu and log(sigma^2)
        self.mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.log_var = nn.Linear(hidden_dims[-1], latent_dim)

        ## flows part
        self.amortized_params_flow = amortized_params_flow
        self.n_flows = n_flows

        if self.amortized_params_flow:
            self.u = nn.Linear(hidden_dims[-1], latent_dim * self.n_flows)
            self.weights = nn.Linear(hidden_dims[-1], latent_dim * self.n_flows)
            self.bias = nn.Linear(hidden_dims[-1], self.n_flows)


    def forward(self, input):
        x = input
        batch_size = x.size(0)
        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        ## now we should compute the mu and log var
        _mu = self.mu(x)
        # _log_var = F.softplus(self.log_var(x))
        _log_var = self.log_var(x)

        ## now we have also to return our z as the reparametrization trick told us
        ## z = mu + sigma * noise, where the noise is sample

        z = reparametrization_trick(_mu, _log_var)

        if self.amortized_params_flow:
            u = self.u(x)
            w = self.weights(x)
            b = self.bias(x)

            # print( u.view(batch_size, self.n_flows, self.z_dims).shape)
            return z, _mu, _log_var, u.view(batch_size, self.n_flows, self.z_dims, 1), w.view(batch_size, self.n_flows, 1, self.z_dims), b.view(batch_size, self.n_flows, 1, 1)

        else:
            return z, _mu, _log_var


## now we have to create the Decoder class
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, input_dim):
        '''

        :param latent_dim: dimension of the latent space (scalar)
        :param hidden_dims: dimensions of the hidden layers (vector)
        :param input_dim: dimension of the input (scalar)
        '''

        super(Decoder, self).__init__()

        # this is kind of symmetric to the encoder, it starts from the latent variables z and it
        # tries to get the original x back

        neurons = [latent_dim, *hidden_dims]
        self.hidden_layers = nn.ModuleList([nn.Linear(neurons[i - 1], neurons[i]) for i in range(1, len(neurons))])
        # self.maxout = Maxout(4)
        # dim_after_maxout = int(hidden_dims[-1] / 4)
        self.reconstruction = nn.Linear(hidden_dims[-1], input_dim)
        self.output_activation = nn.Sigmoid()

    def forward(self, input):
        x = input
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        # print(self.conditional_reconstruction(x).shape)
        return self.output_activation(self.reconstruction(x))


## at this point we have both the encoder and decoder, so we can create the VAE

class VariationalAutoencoderWithFlows(nn.Module):
    def __init__(self,  input_dim, hidden_dims, latent_dim, n_flows, flow_type = PlanarFlow, amortized_params_flow = False):
        '''
        Variational AutoEncoder as described in Kingma and Welling 2014. We have an encoder - decoder
        and we want to learn a meaningful latent representation to being able to reconstruct the input

        :param input_dim: dimension of the input
        :param hidden_dims: dimension of hidden layers #todo: maybe we can differentiate between the encoder and decoder?
        :param latent_dim: dimension of the latent variables
        '''

        super(VariationalAutoencoderWithFlows, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.z_dims = latent_dim

        ## infos about using flows
        self.n_flows = n_flows
        self.flow_type = flow_type
        self.amortized_params_flow = amortized_params_flow

        ## we should create the encoder and the decoder

        if self.amortized_params_flow:
            self.encoder = Encoder(input_dim, hidden_dims, latent_dim, self.amortized_params_flow, self.n_flows)
            self.flows = NormalizingFlows(self.n_flows, self.amortized_params_flow, None)
        else:
            self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
            self.flows = NormalizingFlows(self.n_flows, z_dim = self.z_dims)

        self.decoder = Decoder(latent_dim, list(reversed(hidden_dims)), input_dim)


        self.kl_divergence = 0

        ## we should initialize the weights #
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
    #
    # def _kl_divergence(self, z, q_params, p_params = None):
    #     '''
    #     The function compute the KL divergence between the distribution q_phi(z|x) and the prior p_theta(z)
    #     of a sample z.
    #
    #     KL(q_phi(z|x) || p_theta(z))  = -∫ q_phi(z|x) log [ p_theta(z) / q_phi(z|x) ]
    #                                   = -E[log p_theta(z) - log q_phi(z|x)]
    #
    #     :param z: sample from the distribution q_phi(z|x)
    #     :param q_params: (mu, log_var) of the q_phi(z|x)
    #     :param p_params: (mu, log_var) of the p_theta(z)
    #     :return: the kl divergence KL(q_phi(z|x) || p_theta(z)) computed in z
    #     '''
    #
    #     ## we have to compute the pdf of z wrt q_phi(z|x)
    #     (mu, log_var) = q_params
    #     qz = log_gaussian(z, mu, log_var)
    #     # print('size qz:', qz.shape)
    #     ## we should do the same with p
    #     if p_params is None:
    #         pz = log_standard_gaussian(z)
    #     else:
    #         (mu, log_var) = p_params
    #         pz = log_gaussian(z, mu, log_var)
    #         # print('size pz:', pz.shape)
    #
    #     kl = qz - pz
    #
    #     return kl

    def _kl_divergence_flows(self, z, q_init_params, new_z, logdet_jacobians, p_params = None):
        '''
        The function compute the KL divergence between the distribution q_phi(z|x) and the prior p_theta(z)
        of a sample z.

        KL(q_phi(z|x) || p_theta(z))  = -∫ q_phi(z|x) log [ p_theta(z) / q_phi(z|x) ]
                                      = -E[log p_theta(z) - log q_phi(z|x)]

        :param z: sample from the distribution q_phi(z|x)
        :param q_params: (mu, log_var) of the q_phi(z|x)
        :param p_params: (mu, log_var) of the p_theta(z)
        :return: the kl divergence KL(q_phi(z|x) || p_theta(z)) computed in z
        '''

        ## we have to compute the pdf of z wrt q_phi(z|x)

        (mu, log_var) = q_init_params
        q0 = log_gaussian(z, mu, log_var)
        # print(q0)
        # print('q0 shape', q0.shape)
        # print('log_det_shape', logdet_jacobians.shape)

        # now we have to compute the qz
        qz = q0 - logdet_jacobians
        # print('size qz:', qz.shape)
        ## we should do the same with p
        if p_params is None:
            pz = log_standard_gaussian(new_z)
        else:
            (mu, log_var) = p_params
            pz = log_gaussian(new_z, mu, log_var)
            # print('size pz:', pz.shape)

        return qz, pz

    # ## in case we are using a gaussian prior and a gaussian approximation family
    # def _analytical_kl_gaussian(self, q_params):
    #     '''
    #     Way for computing the kl in an analytical way. This works for gaussian prior
    #     and gaussian density family for the approximated posterior.
    #
    #     :param q_params: (mu, log_var) of the q_phi(z|x)
    #     :return: the kl value computed analytically
    #     '''
    #
    #     (mu, log_var) = q_params
    #     # print(mu.shape)
    #     # print(log_var.shape)
    #     # prova = (log_var + 1 - mu**2 - log_var.exp())
    #     # print(prova.shape)
    #     # print(torch.sum(prova, 1).shape)
    #     # kl = 0.5 * torch.sum(log_var + 1 - mu**2 - log_var.exp(), 1)
    #     kl = 0.5 * torch.sum(log_var + 1 - mu.pow(2) - log_var.exp(), 1)
    #
    #     return kl



    def forward(self, input):
        '''
        Given an input, we want to run the encoder, compute the kl, and reconstruct it

        :param input: an input example
        :return: the reconstructed input
        '''

        # we pass the input through the encoder
        if self.amortized_params_flow:
            z, z_mu, z_log_var, u, w, b = self.encoder(input)
            # we have to process the z through the flows
            new_z, logdet_jacobians = self.flows(z, u, w, b)
        else:
            z, z_mu, z_log_var = self.encoder(input)
            # we have to process the z through the flows
            new_z, logdet_jacobians = self.flows(z, None, None, None)
        # print('original z ', z)

        # we compute the kl
        self.qz, self.pz = self._kl_divergence_flows(z, (z_mu, z_log_var), new_z, logdet_jacobians)
        # self.kl_analytical = self._analytical_kl_gaussian((z_mu, z_log_var))


        # we reconstruct it
        #         print(z
        x_mu = self.decoder(new_z)

        return x_mu


    def sample(self, n_images): # TODO: FIX SAMPLE--> WE ARE SAMPLE Z0 BUT NOT Z_K
        '''
        Method to sample from our generative model

        :return: a sample starting from z ~ N(0,1)
        '''

        z = torch.randn((n_images, self.z_dims), dtype = torch.float)
        # print(z)
        samples =  self.decoder(z)

        return samples












