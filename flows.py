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

    def __init__(self, z_dim):
        '''

        :param z_dim: since we are using it with VAEs this would be the size of the latent.
                      However, flows can be used also in other scenario where the input they
                      get is not the latent variables
        '''
        super(PlanarFlow, self).__init__()
        self.init_sigma = 0.01
        # self.u3 = nn.Parameter(torch.randn(1, 2).normal_(0, init_sigma))
        self.n_features = z_dim
        self.weights = nn.Parameter(torch.randn(1, z_dim).normal_(0, self.init_sigma))
        self.bias = nn.Parameter(torch.zeros(1).normal_(0, self.init_sigma))
        self.u = nn.Parameter(torch.randn(1, z_dim).normal_(0, self.init_sigma))


    def forward(self, z):

        ## we follow the instruction on the Appendix A of the paper:
        ## for the planar flow to be invertible whe using tanh a sufficient
        ## condition is to w^Tu >= -1
        ## --> we have to compute uhat parallel to w
        u_temp = (self.weights @ self.u.t()).squeeze()
        # print('u_temp', u_temp.shape)
        m_u_temp = -1 + F.softplus(u_temp)
        # print('m_u_temp', m_u_temp.shape)

        uhat = self.u + (m_u_temp - u_temp) * (self.weights / (self.weights @ self.weights.t()))
        # print('uhat', uhat.shape)

        z_temp = z @ self.weights.t() + self.bias #F.linear(z, self.weights, self.bias)

        new_z = z + uhat * torch.tanh(z_temp)

        ## now we have to compute psi

        psi = (1 - torch.tanh(z_temp)**2) @ self.weights
        # print('psi', psi.shape)

        det_jac = 1 + psi @ uhat.t() #uhat * psi

        logdet_jacobian = torch.log(torch.abs(det_jac) + 1e-8).squeeze()
        # print(torch.sum(logdet_jacobian, 1).shape)
        # print('log_det_jacobian', logdet_jacobian.shape)

        return new_z, logdet_jacobian # torch.sum(logdet_jacobian, 1)


# class PlanarFlow(nn.Module):
#
#     def __init__(self, z_dim):
#         '''
#
#         :param z_dim: since we are using it with VAEs this would be the size of the latent.
#                       However, flows can be used also in other scenario where the input they
#                       get is not the latent variables
#         '''
#         super(PlanarFlow, self).__init__()
#         self.init_sigma = 0.01
#         # self.u3 = nn.Parameter(torch.randn(1, 2).normal_(0, init_sigma))
#         self.n_features = z_dim
#         # self.weights = nn.Parameter(torch.randn(1, z_dim).normal_(0, self.init_sigma))
#         # self.bias = nn.Parameter(torch.zeros(1).normal_(0, self.init_sigma))
#         # self.u = nn.Parameter(torch.randn(1, z_dim).normal_(0, self.init_sigma))
#
#
#     def forward(self, z, weights, bias, u):
#
#         ## we follow the instruction on the Appendix A of the paper:
#         ## for the planar flow to be invertible whe using tanh a sufficient
#         ## condition is to w^Tu >= -1
#         ## --> we have to compute uhat parallel to w
#         u_temp = (weights @ u.t()).squeeze()
#         # print('u_temp', u_temp.shape)
#         m_u_temp = -1 + F.softplus(u_temp)
#         # print('m_u_temp', m_u_temp.shape)
#
#         uhat = u + (m_u_temp - u_temp) * (self.weights / (self.weights @ self.weights.t()))
#         # print('uhat', uhat.shape)
#
#         z_temp = z @ self.weights.t() + self.bias #F.linear(z, self.weights, self.bias)
#
#         new_z = z + uhat * torch.tanh(z_temp)
#
#         ## now we have to compute psi
#
#         psi = (1 - torch.tanh(z_temp)**2) @ self.weights
#         # print('psi', psi.shape)
#
#         det_jac = 1 + psi @ uhat.t() #uhat * psi
#
#         logdet_jacobian = torch.log(torch.abs(det_jac) + 1e-8).squeeze()
#         # print(torch.sum(logdet_jacobian, 1).shape)
#         # print('log_det_jacobian', logdet_jacobian.shape)
#
#         return new_z, logdet_jacobian # torch.sum(logdet_jacobian, 1)


class NormalizingFlows(nn.Module):

    def __init__(self, z_dims, n_flows = 1, flow_type = PlanarFlow):
        '''

        :param z_dims: dimension of the latent variables
        :param n_flows: how many flows we should use in term of sequence of function f_k (f_k-1(f_k-2(..
        :param flow_type: we have implemented only the Planar Flow, but in case we implement also the radial flow, one can
                          select what type of flows to use
        '''

        super(NormalizingFlows, self).__init__()
        self.z_dims = z_dims
        self.n_flows = n_flows
        self.flow_type = flow_type

        flows_sequence = [self.flow_type(self.z_dims) for _ in range(self.n_flows)]

        self.flows = nn.ModuleList(flows_sequence)

    def forward(self, z):

        # we have to collect all the logdet_jacobian to sum them up in the end
        logdet_jacobians = []
        # i = 0
        for flow in self.flows:
            # i += 1
            # print(i)
            z, logdet_j = flow(z)
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






