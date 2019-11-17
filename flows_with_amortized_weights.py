'''
Implementation of 'Variational Inference with Normalizing Flows' by [Rezende, D. & Mohamed, S. 2015]
'''

import math
import torch
import torch.utils
import torch.utils.data
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import LogNormal
from torch.nn import init
#
class PlanarFlow(nn.Module):

    def __init__(self, amortized_params_flow = False, z_dim = None):
        '''
        We are considering in this case amortized inference of the weights/params. This means that
        instead of learning some weights that we use for all the examples, we learn specific weights for each example.
        In other words, the u, w, b are computed form the input, so we are learning a function that maps each
        imput to
        '''
        super(PlanarFlow, self).__init__()
        self.amortized_params_flow = amortized_params_flow

        if not self.amortized_params_flow:
            self.init_sigma = 0.01
            self.n_features = z_dim
            self.weights = nn.Parameter(torch.randn(1, z_dim).normal_(0, self.init_sigma))
            self.bias = nn.Parameter(torch.zeros(1).normal_(0, self.init_sigma))
            self.u = nn.Parameter(torch.randn(1, z_dim).normal_(0, self.init_sigma))


    def forward(self, zk, u_k, weights_k, bias_k):

        if self.amortized_params_flow:
            '''
            Assumes the following input shapes:
            shape u = (batch_size, z_size, 1)
            shape w = (batch_size, 1, z_size)
            shape b = (batch_size, 1, 1)
            shape z = (batch_size, z_size)
            
            this part is taken from https://github.com/riannevdberg/sylvester-flows/blob/master/models/flows.py
            '''
            # print('z',zk.shape) #
            # print('u', u_k.shape)
            # print('w', weights_k.shape)
            # print('b', bias_k.shape)
            u = u_k
            weights = weights_k
            bias = bias_k

            zk = zk.unsqueeze(2)
            # reparameterize u such that the flow becomes invertible (see appendix paper)
            uw = torch.bmm(weights, u)
            m_uw = -1. + F.softplus(uw)
            w_norm_sq = torch.sum(weights ** 2, dim=2, keepdim=True)
            u_hat = u + ((m_uw - uw) * weights.transpose(2, 1) / w_norm_sq)
            # print('uhat', u_hat.shape)

            # compute flow with u_hat
            wzb = torch.bmm(weights, zk) + bias
            new_z = zk + u_hat * torch.tanh(wzb)
            new_z = new_z.squeeze(2)

            # compute logdetJ
            psi = weights * (1-torch.tanh(wzb)**2)
            log_det_jacobian = torch.log(torch.abs(1 + torch.bmm(psi, u_hat))+ 1e-8)
            logdet_jacobian = log_det_jacobian.squeeze(2).squeeze(1)
            # print(logdet_jacobian.shape)


        else:

            u = self.u
            weights = self.weights
            bias = self.bias

            ## we follow the instruction on the Appendix A of the paper:
            ## for the planar flow to be invertible whe using tanh a sufficient
            ## condition is to w^Tu >= -1
            ## --> we have to compute uhat parallel to w
            u_temp = (weights @ u.t()).squeeze()
            # print('u_temp', u_temp.shape)
            m_u_temp = -1 + F.softplus(u_temp)
            # print('m_u_temp', m_u_temp.shape)

            uhat = u + (m_u_temp - u_temp) * (weights / (weights @ weights.t()))
            print('uhat', uhat.shape)

            z_temp = zk @ weights.t() + bias #F.linear(z, self.weights, self.bias)

            new_z = zk + uhat * torch.tanh(z_temp)

            ## now we have to compute psi

            psi = (1 - torch.tanh(z_temp)**2) @ weights
            print('psi', psi.shape)

            det_jac = 1 + psi @ uhat.t() #uhat * psi

            logdet_jacobian = torch.log(torch.abs(det_jac) + 1e-8).squeeze()
            # print(torch.sum(logdet_jacobian, 1).shape)
            print('log_det_jacobian', logdet_jacobian.shape)

        return new_z, logdet_jacobian # torch.sum(logdet_jacobian, 1)




class NormalizingFlows(nn.Module):

    def __init__(self, n_flows = 1, amortized_params_flow = False, z_dim = None, flow_type = PlanarFlow):
        '''

        :param z_dims: dimension of the latent variables
        :param n_flows: how many flows we should use in term of sequence of function f_k (f_k-1(f_k-2(..
        :param flow_type: we have implemented only the Planar Flow, but in case we implement also the radial flow, one can
                          select what type of flows to use
        '''

        super(NormalizingFlows, self).__init__()
        self.n_flows = n_flows
        self.flow_type = flow_type
        self.amortized_params_flow = amortized_params_flow
        self.z_dim = z_dim

        flows_sequence = [self.flow_type(self.amortized_params_flow, self.z_dim) for _ in range(self.n_flows)]

        self.flows = nn.ModuleList(flows_sequence)

    def forward(self, z, u, w, b):

        # we have to collect all the logdet_jacobian to sum them up in the end
        logdet_jacobians = []
        # i = 0
        # print('u')
        # print(u)
        # print(u.shape)
        for k, flow in enumerate(self.flows):
            # i += 1
            # print(i)
            # print(self.amortized_params_flow)
            if self.amortized_params_flow:
                z, logdet_j = flow(z, u[:, k, :, :], w[:, k, :, :], b[:, k, :, :])
            else:
                z, logdet_j = flow(z, None, None, None)
            # print(logdet_j.shape)
            logdet_jacobians.append(logdet_j)

        z_k = z
        # print('final_logdet_jacobian', logdet_jacobians)
        logdet_jacobians = torch.stack(logdet_jacobians, dim=1)
        # print(logdet_jacobians.shape)
        # print(torch.sum(logdet_jacobians, 1).shape)
        # print('new_z', z_k.shape)
        # print('we sum them')
        # print(torch.sum(logdet_jacobians, 1))
        return z_k, torch.sum(logdet_jacobians, 1)






