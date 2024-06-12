#!/usr/bin/env/python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.distributions import Normal
from scipy.special import psi, gamma
from scipy.misc import derivative


class MCElboMF(torch.nn.Module):
    def __init__(self, nMC, x_dim, param_dim, f_dim, err_cs, Elab_cs, f_sigmas, use_theory_cov, data_train):
        """ MCElbo is a child of torch.nn.Module class that implements the model likelihood and forward function for the VI
            approximation
            
            nMc: Number of samples for MC approximation of ELBO (typically 5 - 10)
            x_dim: Dimension of known experimental inputs (for example Z and N)
            param_dim: Dimension of calibration parameters
            f_dim: Dimension of normalization parameters
            err_cs: Experimental errors
            Elab_cs: Experimental energies
            f_sigmas: Uncertainties in the normalization parameters
            use_theory_cov: Boolean flag to determine if the theoretical covariance matrix should be used
        """
        super(MCElboMF, self).__init__()
        self.root_path = './'
        self.err_cs = err_cs
        self.Elab_cs = Elab_cs
        self.n = len(err_cs)
    
        self.nMC = nMC
        self.x_dim = x_dim
        self.param_dim = param_dim #bit redundant to have both
        self.erp_dim = param_dim
        self.f_dim = f_dim
        self.softplus = torch.nn.Softplus() # transformation of the input x \in R tolog(1+exp(x)) (avoids constrained opt.)
         
        ####### VI inits ######    
        
        ### Prior distributions (assumed independent gaussian)
        
        gauss_prior_params = np.array([[0.025, 0.015], [0.8, 0.4], [13.84, 1.63], [0.0, 1.6], [12.59, 1.85], [0.0, 1.6]]) # center, width
        f_mus = np.ones(self.f_dim)
        
        self.prior_theta_m = torch.tensor(np.concatenate([gauss_prior_params[:,0],f_mus]).reshape(1,self.param_dim + self.f_dim))
        self.prior_theta_s = torch.tensor(np.concatenate([gauss_prior_params[:,1],f_sigmas]).reshape(1,self.param_dim + self.f_dim))
        
        ### Variational parameters for the mean-field (fully factorized) variational posterior
        self.q_theta_m = torch.nn.Parameter(self.prior_theta_m.detach().clone()) # innitial values
        self.q_theta_s = torch.nn.Parameter(torch.expm1(self.prior_theta_s.detach().clone()).log())
        
        ### Current parameter state place holder
        self.params = self.prior_theta_m.detach().clone()
        
        # If likelihood is gaussian parametrized with precision (fixed for now)
        #self.precision = torch.FloatTensor((1))
        #self.precision.data.fill_(1/ (2.97 ** 2))
        
        # If likelihood is gaussian parametrized with std (fixed for now)
        #self.s = torch.FloatTensor((1))
        #self.s.data.fill_(1/ (2.97 ** 2))
        
        #########################################
        ############ Scatering inits #############
        ##########################################
        
        self.Zb = 2 # Charge number
        self.Zt = 2 # Charge number
        self.mb = 2809.43 # MeV - 3He
        self.mt = 3728.42 # MeV - alpha
        self.mu = (self.mb * self.mt) / (self.mb + self.mt) # Reduced mass

        # Set useful constants and variables
        self.alpha = 1 / 137.036 # dimless (fine structure alpha_em)
        self.h_bar_c = 197.327 # MeV fm
        
        ##############################################################################
                    # Define the 'utility' functions as lambda functions #
        ##############################################################################
        
        # Square root to handle negative values
        self.sqrt = lambda x: np.sqrt(x) if x >= 0 else 1.0j * np.sqrt(np.abs(x))
        self.sqrt = np.vectorize(self.sqrt)

        # Cotangent to handle arguments in degrees
        self.cot = lambda theta_deg: (np.cos(theta_deg * (np.pi / 180)) /
                        np.sin(theta_deg * (np.pi / 180)))

        # Energy from lab frame to CM frame
        self.ECM = lambda Elab: ((2 * self.mt) / (self.mt + self.mb +
                        self.sqrt(((self.mt + self.mb)**2) +
                        2 * self.mt * Elab))) * Elab  #MeV
        
        # Energy from CM frame to lab frame
        self.ELAB = lambda Ecm: ((Ecm + (2 * (self.mt + self.mb))) /
                        (2.0 * self.mt)) * Ecm  #MeV 

        # Obtain kc as a function of Elab
        self.kc = lambda Elab: self.Zb * self.Zt * self.alpha * (
                        (self.mu + self.ECM(Elab)) / self.h_bar_c)  # fm^-1
        
        # Obtain k as a function of Elab
        self.k = lambda Elab: (1 / self.h_bar_c) * self.sqrt(
                        ((self.mu + self.ECM(Elab))**2) - self.mu**2)
        
                # Obtain eta as a function of Elab
        self.eta = lambda Elab: self.kc(Elab) / self.k(Elab)

        # Obtain H as a function of Elab
        self.H = lambda Elab: ((psi((1.j) * self.eta(Elab))) +
                            (1 / (2.j * self.eta(Elab))) -
                            np.log(1.j * self.eta(Elab)))
        
         # Define h as the real part of H as a function of Elab
        self.h = lambda Elab: self.H(Elab).real

        # Obtain C0_2 as a function of Elab
        self.C0_2 = lambda Elab: (2 * np.pi * self.eta(Elab)) / (
                            np.exp(2 * np.pi * self.eta(Elab)) - 1)
        
               # Obtain C1_2 as a function of Elab
        self.C1_2 = lambda Elab: (1 / 9) * (1 + self.eta(Elab)**2) * self.C0_2(Elab)

        # Obtain C2_2 as a function of Elab
        self.C2_2 = lambda Elab: (1 / 100) * (4 + self.eta(Elab)**2) * self.C1_2(Elab)

        # Obtain C3_2 as a function of Elab
        self.C3_2 = lambda Elab: (1 / 441) * (9 + self.eta(Elab)**2) * self.C2_2(Elab)

        # Derivative of H with respect to eta
        self.H_prime_eta = lambda eta: derivative(lambda x: (psi(1.j * x) + 1 / (2 * 1.j * x) - np.log(1.j * x)), eta, dx = 1e-10)


        # # # Save / Read in various files for values # # #
        self.kvalue_cs = torch.tensor(self.k(self.Elab_cs))
        self.kcvalue_cs = torch.tensor(self.kc(self.Elab_cs))
        self.etavalue_cs = torch.tensor(self.eta(self.Elab_cs))
        self.Hvalue_cs = torch.tensor(self.H(self.Elab_cs))
        self.Hvalue_cs_real = torch.tensor(self.Hvalue_cs.real)
        self.Hvalue_cs_imag = torch.tensor(self.Hvalue_cs.imag)
        self.C0_2value_cs = torch.tensor(self.C0_2(self.Elab_cs))
        self.C1_2value_cs = torch.tensor(self.C1_2(self.Elab_cs))
            
        ##############################################################################
                            # Define the binding energy of the 7Be #
        ##############################################################################
        # Binding energy of 7Be in 3/2+ and 1/2- channels
        E_plus = torch.tensor(1.5866) # MeV
        E_minus = torch.tensor(1.1575) # MeV

        # Binding momenta of the 3/2+ and 1/2- channels
        self.gamma1_plus = np.sqrt(2 * self.mu * E_plus) / self.h_bar_c
        self.gamma1_minus = np.sqrt(2 * self.mu * E_minus) / self.h_bar_c

        # Get the kc's
        self.kc_plus = (self.Zb * self.Zt * self.alpha * (self.mu - E_plus)) / self.h_bar_c
        self.kc_minus = (self.Zb * self.Zt * self.alpha * (self.mu - E_minus)) / self.h_bar_c

        # Define the Sommerfeld parameters
        self.eta_B_plus = self.kc_plus / self.gamma1_plus
        self.eta_B_minus = self.kc_minus / self.gamma1_minus

        # Define the H's
        self.H_plus = psi(self.eta_B_plus) + (1 / (2 * self.eta_B_plus)) - np.log(self.eta_B_plus)
        self.H_minus = psi(self.eta_B_minus) + (1 / (2 * self.eta_B_minus)) - np.log(self.eta_B_minus)
        self.H_prime_plus = self.H_prime_eta(-self.eta_B_plus * (1.j))
        self.H_prime_minus = self.H_prime_eta(-self.eta_B_minus * (1.j))


        ##############################################################################
                            # Andrius - set bounds  + covariance 
        ##############################################################################
        param_bounds = np.array([[-0.02, 0.06], [-3, 3], [5.0, 25.0], [-6, 6], [5.0, 25.0], [-6, 6]]) # lower, upper
        f_bounds = np.array([[0.0, 2.0] for i in range(self.f_dim)]) # lower, upper
        self.bounds = torch.tensor(np.concatenate([param_bounds, f_bounds]))

        # Set up the covariance matrix
        self.data_train = data_train
        cov_expt_matrix = np.diag(self.err_cs**2)
        cov_theory_matrix = self.cov_theory()
        if use_theory_cov:
            cov_matrix = cov_expt_matrix + cov_theory_matrix
        else:
            cov_matrix = cov_expt_matrix
        self.inv_cov_matrix = torch.linalg.inv(torch.tensor(cov_matrix))
        self.Q_rest = np.sum(np.log(1.0 / np.sqrt(2.0 * np.pi * np.diag(cov_matrix))))

    def cov_theory(self):
        """
        Computes the covariance theory matrix.
        """
        y0 = np.reshape(self.data_train.data[:, 13], (1, len(self.data_train.data[:, 13])))
        c_rms = 0.70
        Lambda = 200.0 / self.h_bar_c  # fm^-1

        # Convert theta to radians
        theta_rad = self.data_train.data[:, 1] * (np.pi / 180)

        # Get the momentum transfer
        q = 2.0 * self.kvalue_cs * np.sin(theta_rad / 2.0)
        Q = ((np.array([max(q[t], self.data_train.data[:, 6][t])
                    for t in range(len(theta_rad))])) / Lambda)  # dimless
        Q = np.reshape(Q, (1, len(Q)))
        K = 2.0  # N3LO is taken as dominant theoretical error
        cov_theory_cs = (c_rms * y0 * (Q**(K + 1))).transpose() @ (
            c_rms * y0 * (Q**(K + 1))) / (1 - Q.transpose() @ Q)
        return cov_theory_cs

    def __str__(self):
        return f"VI ELBO for a model with {self.param_dim} calibration parameters, {self.x_dim} dimensional input. "\
                + f"{self.nMC} MC samples are used for MC approximation."
        
    def cs_theory_bsc_in_C(self, inputs, params):
        """
        # inputs are teta_cs
        This method utilizes the existence of a bound state and also the asymptotic normalization coefficients
        to compute the theoretical cross section at the energies and angles where we have data.

        NOTE: C1+/- is really (C1+/-)^2 since we are sampling C1 squared
        """
        # Extract the parameters from params
        A0, r0, C1plus, P1plus, C1minus, P1minus = params # this is first time we get params

        # Convert theta to radians
        theta_rad = inputs * (np.pi / 180)

        # Compute r1plus and r1minus based on the relation
        r1plus = torch.real(-(2 * self.gamma1_plus**2 * torch.exp(torch.special.gammaln(2 + self.eta_B_plus))**2) / C1plus + (P1plus * self.gamma1_plus**2) + (
            4 * self.kc_plus * self.H_plus) + (
            2.j * self.kc_plus * self.eta_B_plus * (1 - self.eta_B_plus**2) * self.H_prime_plus))
        r1minus = torch.real(-(2 * self.gamma1_minus**2 * torch.exp(torch.special.gammaln(2 + self.eta_B_minus))**2) / C1minus + (P1minus * self.gamma1_minus**2) + (
            4 * self.kc_minus * self.H_minus) + (
            2.j * self.kc_minus * self.eta_B_minus * (1 - self.eta_B_minus**2) * self.H_prime_minus))

        # Compute A1plus and A1minus based on the bound state condition
        A1plus = - (r1plus * self.gamma1_plus**2) / 2 + (0.25 * P1plus * self.gamma1_plus**4) + (
            2 * self.kc_plus * (self.gamma1_plus**2 - self.kc_plus**2)) * self.H_plus
        A1minus = - (r1minus * self.gamma1_minus**2) / 2 + (0.25 * P1minus * self.gamma1_minus**4) + (
            2 * self.kc_minus * (self.gamma1_minus**2 - self.kc_minus**2)) * self.H_minus

        # Set up the necessary Legendre polynomials
        P_1 = torch.cos(theta_rad)

        # Phase terms
        alpha_1 = torch.arcsin((self.etavalue_cs / torch.sqrt(1 + torch.square(self.etavalue_cs))))
        phase_1 = torch.cos(2 * alpha_1) + (1.j) * torch.sin(2 * alpha_1)

        K_0 = (1 / (2 * self.kcvalue_cs)) * (-A0 + 0.5 * r0 * self.kvalue_cs**2)
        K_1_plus = (1 / (2 * self.kcvalue_cs**3)) * (-A1plus + 
                0.5 * r1plus * self.kvalue_cs**2 + 
                    0.25 * P1plus * self.kvalue_cs**4)
        K_1_minus = (1 / (2 * self.kcvalue_cs**3)) * (-A1minus + 
                0.5 * r1minus * self.kvalue_cs**2 + 
                    0.25 * P1minus * self.kvalue_cs**4)
        
        ERE_0 = (2 * self.kcvalue_cs * (K_0 - self.Hvalue_cs)) / self.C0_2value_cs
        ERE_1_plus = (2 * self.kcvalue_cs / (9 * self.C1_2value_cs)) * (
            self.kcvalue_cs**2 * K_1_plus - 
                (self.kcvalue_cs**2 + self.kvalue_cs**2) * self.Hvalue_cs)
        ERE_1_minus = (2 * self.kcvalue_cs / (9 * self.C1_2value_cs)) * (
            self.kcvalue_cs**2 * K_1_minus - 
                (self.kcvalue_cs**2 + self.kvalue_cs**2) * self.Hvalue_cs)

        # Compute the amplitude of each of the components
        # Rutherford
        f_r = (-self.etavalue_cs / (2 * self.kvalue_cs)) * (
            1 / np.sin(theta_rad / 2)**2)**(1 + (1.j) * self.etavalue_cs)
        
        # Coulomb
        f_c = f_r + (1 / ERE_0) + (self.kvalue_cs**2 * phase_1 * P_1) * (
            (2 / ERE_1_plus) + (1 / ERE_1_minus))

        # Interaction
        f_i = self.kvalue_cs**2 * phase_1 * torch.sin(theta_rad) * (
            (1 / ERE_1_minus) - (1 / ERE_1_plus))

        sigma = 10 * (torch.abs(f_c)**2 + torch.abs(f_i)**2)
        sigma_R = 10 * torch.abs(f_r)**2
        sigma_ratio = sigma / sigma_R
        return sigma_ratio    
    
    def chi_squared(self, theory, expt, norm):
        """
        """
        #print(norm.shape, theory.shape, expt.shape)
        r = norm * theory - expt
        chi_2 = r @ self.inv_cov_matrix @ r
        return chi_2, r
    
    def log_likelihood(self, inputs, exp):
        """
        Determines the likelihood of a set of parameters given the data.
        """
        parameters = self.params[0]

        # Unpack the erps and normalization parameters
        params = parameters[:self.erp_dim]
        params_f = parameters[self.erp_dim:]

        # Get the normalizations and theory
        norm = params_f[self.data_train.data[:, 4].astype(int)]
        theory = self.cs_theory_bsc_in_C(inputs, params)

        # Compute the chi squared
        chi2 = self.chi_squared(theory, exp, norm)[0]

        # Compute the likelihood
        log_L = -0.5 * chi2 + self.Q_rest
        return log_L

    # def log_likelihood(self, inputs, exp):
    #     """
    #     Determines the likelihood of a set of parameters given the data.
    #     """
    #     parameters = self.params[0]

    #     # Unpack the erps and normalization parameters
    #     params = parameters[:self.erp_dim]
    #     params_f = parameters[self.erp_dim:]

    #     # If the parameters are within the bounds, compute the log likelihood
    #     if np.logical_and(self.bounds[:, 0] <= parameters, parameters <= self.bounds[:, 1]).all():
    #         # Get the normalizations and theory
    #         norm = params_f[self.data_train.data[:, 4].astype(int)]
    #         theory = self.cs_theory_bsc_in_C(inputs, params)

    #         # Compute the chi squared
    #         chi2 = self.chi_squared(theory, exp, norm)[0]

    #         # Compute the likelihood
    #         log_L = -0.5 * chi2 + self.Q_rest
    #         return log_L
    #     else:
    #         return torch.tensor(float('-inf'))
    
    def log_Normal(self, param, m, s):
        """Computes log likelihood of normal density
        Args:
            param: 1 X K dimensional tensor
            m: 1 X K dimensional mean tensor
            s: 1 X K dimensional std tensor
        """
        interm__ = ((param - m) / s).pow(2)
        log_prob = - len(param) * np.log(2 * np.pi) / 2 - torch.log(s).sum() - 0.5 * interm__.sum()
        return log_prob.reshape(1,1)
        
    def q_log(self):
        """Variational family log likelihood. Gaussian."""
        log_lkl = self.log_Normal(self.params, self.q_theta_m, self.softplus(self.q_theta_s))  
        return log_lkl
        
    def pri_log(self):
        """Prior distribution log likelohood. Gaussian"""
        log_lkl = self.log_Normal(self.params, self.prior_theta_m, self.prior_theta_s) 
        return log_lkl
    
    def generate_sample(self, n_var = 2):
        """Used for sampling from a variational family"""
        return Normal(0,1).sample((self.nMC,n_var,1))
    
    def sample_reparam_normal(self, param, m, s):
        """Reparametrization trick"""
        return param.mul(self.softplus(s.T)).add(m.T)
    
    #def data_likelihood(self, x, y):
    #    """Returns data gaussian likelihood for one MC sample. Precision parametrization
    #    
    #        x: inputs (assumed to be n x x_dim tensor)
    #        y: experimental observations (assumed to be a row tensor of size n)
    #    """
    #    y_fit = y  #This is a placeholder, y_fit is the model fit 
    #    log_likelihood = (torch.log(self.precision) - np.log(2 * np.pi)) * 0.5 * self.n  - 0.5 * self.precision * torch.sum(torch.pow(y-y_fit, 2))    
    #    return log_likelihood
    
    def compute_elbo_loop(self, x, y):     
        
        z = self.generate_sample(n_var = self.param_dim + self.f_dim)
        theta = self.sample_reparam_normal(z, self.q_theta_m, self.q_theta_s)
       
        loss = torch.randn(1,1)
        loss.data.fill_(0.0)
        
        for i in range(self.nMC):
            self.params = theta[i].T
            
            q_likelihood = self.q_log()
            prior_likelihood = self.pri_log()
            data_likelihood = self.log_likelihood(x, y)
            loss += data_likelihood + prior_likelihood - q_likelihood
        return loss / self.nMC
    
    def ModelString(self):
        return f"theta mean: {self.q_theta_m} \n theta std {self.softplus(self.q_theta_s)}"