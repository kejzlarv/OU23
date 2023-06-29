#!/usr/bin/env/python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.distributions import Normal
from scipy.special import psi, gamma
from scipy.misc import derivative

class MCElboMF(torch.nn.Module):
    def __init__(self, nMC, x_dim, param_dim, f_dim, err_cs, Elab_cs, f_sigmas,
                 recompute_values = True, which_data = "both", barnard_Elab = None, l_som = None):
        """ MCElbo is a child of torch.nn.Module class that implements the model likelihood and forward function for the VI
            approximation with mean-field (fully factorized) gaussian variational family.
            
            Arguments:
            ----------
            
            nMc: int
            Number of samples for MC approximation of ELBO (typically 5 - 10)
            
            x_dim: int
            Dimension of known experimental inputs (for example Z and N) - current implementation does not make any use of this information
            
            param_dim: int
            Dimension of calibration parameters
            
            f_dim: int 
            Dimension of the normalizing factors
            
            err_cs: numpy.array
            Array of experimental uncertainties 
            
            Elab_cs: numpy.array
            Array of Elab data producted by the DataLoader from scattering_data.py
            
            f_sigmas: numpy.array
            Array of prior standard deviations for the normalizing factors
            
            recompute_values: bool
            Intermediate computations to be carried again?
            
            which_data: string
            Determines which data to look at
            
            barnard_Elab: numpy.array
            Array of barnard_Elab data producted by the DataLoader from scattering_data.py. Only if which_data is "both" or "barnard"
            
            l_som: numpy.array
            Array of l_som data producted by the DataLoader from scattering_data
        """
        super(MCElboMF, self).__init__()
        ##############################################################################
                    # Set the initial constants and useful variables
        ##############################################################################
        self.l_som = l_som
        self.barnard_Elab = barnard_Elab
        self.which_data = which_data
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
        f_mus = np.ones(self.f_dim) # means for the normalizing factors
        
        self.prior_theta_m = torch.tensor(np.concatenate([gauss_prior_params[:,0],f_mus]).reshape(1,self.param_dim + self.f_dim))
        self.prior_theta_s = torch.tensor(np.concatenate([gauss_prior_params[:,1],f_sigmas]).reshape(1,self.param_dim + self.f_dim))
        
        ### Variational parameters for the mean-field (fully factorized) variational posterior
        self.q_theta_m = torch.nn.Parameter(self.prior_theta_m.detach().clone()) # innitial values
        self.q_theta_s = torch.nn.Parameter(torch.expm1(self.prior_theta_s.detach().clone()).log())
        
        ### Current parameter state place holder
        self.params = self.prior_theta_m.detach().clone()
        
        #########################################
        ############ Scatering inits #############
        ##########################################
        
         # # # Do statistics stuff # # #
        cov_expt_matrix = np.diag(self.err_cs**2)
        cov_matrix = cov_expt_matrix # + cov_theory_matrix
        ####
        
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
        if recompute_values:
            # Save files and then immediately read them back in
            save_k = np.savetxt('kvalue_cs.txt', self.k(self.Elab_cs), delimiter = ',')
            self.kvalue_cs = torch.tensor(np.loadtxt(self.root_path + 'kvalue_cs.txt'))
            save_kc = np.savetxt('kcvalue_cs.txt', self.kc(self.Elab_cs), delimiter = ',')
            self.kcvalue_cs = torch.tensor(np.loadtxt(self.root_path + 'kcvalue_cs.txt'))
            save_eta = np.savetxt('etavalue_cs.txt', self.eta(self.Elab_cs), delimiter = ',')
            self.etavalue_cs = torch.tensor(np.loadtxt(self.root_path + 'etavalue_cs.txt'))
            save_H_real = np.savetxt('Hvalue_cs_real.txt', np.real(self.H(self.Elab_cs)), delimiter = ',')
            self.Hvalue_cs_real = torch.tensor(np.loadtxt(self.root_path + 'Hvalue_cs_real.txt'))
            save_H_imag = np.savetxt('Hvalue_cs_imag.txt', np.imag(self.H(self.Elab_cs)), delimiter = ',')
            self.Hvalue_cs_imag = torch.tensor(np.loadtxt(self.root_path + 'Hvalue_cs_imag.txt'))
            self.Hvalue_cs = self.Hvalue_cs_real + 1.j * self.Hvalue_cs_imag
            save_C0_2 = np.savetxt('C0_2value_cs.txt', self.C0_2(self.Elab_cs), delimiter = ',')
            self.C0_2value_cs = torch.tensor(np.loadtxt(self.root_path + 'C0_2value_cs.txt'))
            save_C1_2 = np.savetxt('C1_2value_cs.txt', self.C1_2(self.Elab_cs), delimiter = ',')
            self.C1_2value_cs = torch.tensor(np.loadtxt(self.root_path + 'C1_2value_cs.txt'))
            #save_cs_LO = np.savetxt('cs_LO_values.txt', self.cs_LO(), delimiter = ',')
            #self.cs_LO_values = np.loadtxt(self.root_path + 'cs_LO_values.txt')
            save_inv_cov = np.savetxt('inv_cov_matrix.txt', (np.linalg.inv(cov_matrix)).flatten(), delimiter = ',')
            self.inv_cov_matrix = torch.tensor((np.loadtxt('inv_cov_matrix.txt')).reshape((self.n, self.n)))
            #save_cov_theory = np.savetxt('cov_theory_matrix.txt', self.cov_theory().flatten(), delimiter = ',')
            #self.cov_theory_matrix = (np.loadtxt('cov_theory_matrix.txt')).reshape((self.n, self.n))
            self.Q_rest = torch.tensor(np.sum(np.log(1.0 / np.sqrt(2.0 * np.pi * np.diag(cov_matrix)))))
        else:
            # Just read in the files
            self.kvalue_cs = torch.tensor(np.loadtxt(self.root_path + 'kvalue_cs.txt'))
            self.kcvalue_cs = torch.tensor(np.loadtxt(self.root_path + 'kcvalue_cs.txt'))
            self.etavalue_cs = torch.tensor(np.loadtxt(self.root_path + 'etavalue_cs.txt'))
            self.Hvalue_cs_real = torch.tensor(np.loadtxt(self.root_path + 'Hvalue_cs_real.txt'))
            self.Hvalue_cs_imag = torch.tensor(np.loadtxt(self.root_path + 'Hvalue_cs_imag.txt'))
            self.Hvalue_cs = self.Hvalue_cs_real + 1.j * self.Hvalue_cs_imag
            self.C0_2value_cs = torch.tensor(np.loadtxt(self.root_path + 'C0_2value_cs.txt'))
            self.C1_2value_cs = torch.tensor(np.loadtxt(self.root_path + 'C1_2value_cs.txt'))
            #self.cs_LO_values = np.loadtxt(self.root_path + 'cs_LO_values.txt')
            self.inv_cov_matrix = torch.tensor((np.loadtxt('inv_cov_matrix.txt')).reshape((self.n, self.n)))
            #self.cov_theory_matrix = (np.loadtxt('cov_theory_matrix.txt')).reshape((len(self.cs_data), len(self.cs_data)))
            self.Q_rest = torch.tensor(np.sum(np.log(1.0 / np.sqrt(2.0 * np.pi * np.diag(cov_matrix)))))
            
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
        
    def __str__(self):
        return f"VI ELBO for a model with {self.param_dim} calibration parameters, {self.x_dim} dimensional input. "\
                + f"{self.nMC} MC samples are used for MC approximation."
        
    def cs_theory_bsc_in_C(self, inputs, params):
        """
        This method utilizes the existence of a bound state and also the asymptotic normalization coefficients
        to compute the theoretical cross section at the energies and angles where we have data.

        NOTE: C1+/- is really (C1+/-)^2 since we are sampling C1 squared
        
        Arguments:
        ----------
        
        inputs: torch.tensor
        Tensor of theta_cs values
        
        params: torch.tensor
        Tensor combined calibration parameters and normalizing factors [parameters, factors]  
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
         Computes chi-square value. This is an intermediate computatin for log_likelihood
        
        Arguments:
        ----------
        
        theory: torch.tensor
        Tensor of theoretical vales of an observable
        
        expt: torch.tensor
        Tensor of experimental observations
        
        norm: torch.tensor
        Tensor of normalizign factors
        """
        #print(norm.shape, theory.shape, expt.shape)
        r = norm * theory - expt
        chi_2 = r @ self.inv_cov_matrix @ r
        return chi_2, r
    
    def log_likelihood(self, inputs, exp):
        """
        Evaluates the log_likelihood for a given set of experimental inputs and experimental observations.
        
        Arguments:
        ----------
        
        inputs: torch.tensor
        Tensor of theta_cs values
        
        exp: torch.tensor
        Tensor of experimental observations
        """
        parameters = self.params[0]

        # Unpack the erps and normalization parameters
        params = parameters[:self.erp_dim]
        params_f = parameters[self.erp_dim:]

        # If the parameters are within the bounds, compute the log likelihood
        if self.which_data == 'barnard':
            norm_barnard = params_f[-1].repeat(len(self.barnard_Elab))
            norm_som = torch.tensor([])
        if self.which_data == 'som':
            norm_barnard = torch.tensor([])
            norm_som = torch.tensor([])
            for l_int in self.l_som:
                norms = torch.cat([params_f[j].repeat(l_int[j]) for j in range(0, l_int.shape[0])])
                norm_som = torch.cat([norm_som, norms])
        if self.which_data == 'both':
            norm_barnard = params_f[-1].repeat( len(self.barnard_Elab))
            norm_som = torch.tensor([])
            for l_int in self.l_som:
                norms = torch.cat([params_f[j].repeat( l_int[j]) for j in range(0, l_int.shape[0])])
                #print(norms)
                norm_som = torch.cat([norm_som, norms])

        # Set the normalizations, theory values, and the exp values
        norm = torch.cat([norm_som, norm_barnard])
        theory = self.cs_theory_bsc_in_C(inputs,  params)
        experiment = exp

        # Compute the chi squared
        chi2 = self.chi_squared(theory, experiment, norm)[0]

        # Compute the likelihood
        log_L = -0.5 * chi2 + self.Q_rest
        return log_L
    
    def log_Normal(self, param, m, s):
        """Computes log likelihood of normal density
        
        Arguments:
        ----------
        
            param: 1 X K dimensional tensor
            m: 1 X K dimensional mean tensor
            s: 1 X K dimensional std tensor
        """
        interm__ = ((param - m) / s).pow(2)
        log_prob = - len(param) * np.log(2 * np.pi) / 2 - torch.log(s).sum() - 0.5 * interm__.sum()
        return log_prob.reshape(1,1)
        
    def q_log(self):
        """Variational family log likelihood. Gaussian fully factorized."""
        log_lkl = self.log_Normal(self.params, self.q_theta_m, self.softplus(self.q_theta_s))  
        return log_lkl
        
    def pri_log(self):
        """Prior distribution log likelohood. Gaussian"""
        log_lkl = self.log_Normal(self.params, self.prior_theta_m, self.prior_theta_s) 
        return log_lkl
    
    def generate_sample(self, n_var = 2):
        """
        Generate self.nMC from a standard multivariate normal distributino
        
        Arguments:
        ----------
        
        n_var: int
        Number of dimensions      
        """
        return Normal(0,1).sample((self.nMC,n_var,1))
    
    def sample_reparam_normal(self, param, m, s):
        """
        Reparametrization trick.
        
        Arguments:
        ----------
        
            param: 1 X K dimensional tensor
            m: 1 X K dimensional mean tensor
            s: 1 X K dimensional std tensor
        
        """
        return param.mul(self.softplus(s.T)).add(m.T)
    
    def compute_elbo_loop(self, x, y):
        """
        Computes MC ELBO estimate based on the log_likelihood, q_log, and pri_log methods. 
        
        Arguments:
        ----------
        
        x: torch_tensor
        Tensor of theta_cs values
        
        y: torch_tensor
        Tensor of experimental observations
        """
        
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
    
    

class MCElboFG(torch.nn.Module):
    def __init__(self, nMC, x_dim, param_dim, f_dim, err_cs, Elab_cs, f_sigmas,
                 recompute_values = True, which_data = "both", barnard_Elab = None, l_som = None):
        """ MCElbo is a child of torch.nn.Module class that implements the model likelihood and forward function for the VI
            approximation with gaussian variational family (full covariance).
            
            Arguments:
            ----------
            
            nMc: int
            Number of samples for MC approximation of ELBO (typically 5 - 10)
            
            x_dim: int
            Dimension of known experimental inputs (for example Z and N) - current implementation does not make any use of this information
            
            param_dim: int
            Dimension of calibration parameters
            
            f_dim: int 
            Dimension of the normalizing factors
            
            err_cs: numpy.array
            Array of experimental uncertainties 
            
            Elab_cs: numpy.array
            Array of Elab data producted by the DataLoader from scattering_data.py
            
            f_sigmas: numpy.array
            Array of prior standard deviations for the normalizing factors
            
            recompute_values: bool
            Intermediate computations to be carried again?
            
            which_data: string
            Determines which data to look at
            
            barnard_Elab: numpy.array
            Array of barnard_Elab data producted by the DataLoader from scattering_data.py. Only if which_data is "both" or "barnard"
            
            l_som: numpy.array
            Array of l_som data producted by the DataLoader from scattering_data
        """
        
        ##############################################################################
                    # Set the initial constants and useful variables
        ##############################################################################
        super(MCElboFG, self).__init__()
        self.l_som = l_som
        self.barnard_Elab = barnard_Elab
        self.which_data = which_data
        self.root_path = './'
        self.err_cs = err_cs
        self.Elab_cs = Elab_cs
        self.n = len(err_cs)
        
        self.up_i = np.triu_indices(param_dim + f_dim, k =1)
        self.diag_i = (np.array(range(param_dim + f_dim)),np.array(range(param_dim + f_dim)))
        
        self.nMC = nMC
        self.x_dim = x_dim
        self.param_dim = param_dim #bit redundant to have both
        self.erp_dim = param_dim
        self.f_dim = f_dim
        self.softplus = torch.nn.Softplus() # transformation of the input x \in R tolog(1+exp(x)) (avoids constrained opt.)
         
        ####### VI inits ######    
        
        ### Prior distributions (assumed independent gaussian)
        
        #Bounds for truncated normals
        
        #Prior means and sds
        gauss_prior_params = np.array([[0.025, 0.015], [0.8, 0.4], [13.84, 1.63], [0.0, 1.6], [12.59, 1.85], [0.0, 1.6]]) # center, width
        f_mus = np.ones(self.f_dim)
        
        self.prior_theta_m = torch.tensor(np.concatenate([gauss_prior_params[:,0],f_mus]).reshape(1,self.param_dim + self.f_dim))
        self.prior_theta_s = torch.tensor(np.concatenate([gauss_prior_params[:,1],f_sigmas]).reshape(1,self.param_dim + self.f_dim))
        
        ### Variational parameters for the mean-field (fully factorized) variational posterior
        self.q_theta_m = torch.nn.Parameter(self.prior_theta_m.detach().clone()) # innitial values
        self.q_theta_s = torch.nn.Parameter(torch.expm1(self.prior_theta_s.detach().clone()).log())
        self.q_theta_c = torch.nn.Parameter(torch.rand(1,  int((param_dim + f_dim) * (param_dim + f_dim + 1)/2 - (param_dim + f_dim)), dtype=torch.float64))  # corr_params
        
        ### Current parameter state place holder
        self.params = self.prior_theta_m.detach().clone()
        
        
        #########################################
        ############ Scatering inits #############
        ##########################################
        
         # # # Do statistics stuff # # #
        cov_expt_matrix = np.diag(self.err_cs**2)
        cov_matrix = cov_expt_matrix # + cov_theory_matrix
        ####
        
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
        if recompute_values:
            # Save files and then immediately read them back in
            save_k = np.savetxt('kvalue_cs.txt', self.k(self.Elab_cs), delimiter = ',')
            self.kvalue_cs = torch.tensor(np.loadtxt(self.root_path + 'kvalue_cs.txt'))
            save_kc = np.savetxt('kcvalue_cs.txt', self.kc(self.Elab_cs), delimiter = ',')
            self.kcvalue_cs = torch.tensor(np.loadtxt(self.root_path + 'kcvalue_cs.txt'))
            save_eta = np.savetxt('etavalue_cs.txt', self.eta(self.Elab_cs), delimiter = ',')
            self.etavalue_cs = torch.tensor(np.loadtxt(self.root_path + 'etavalue_cs.txt'))
            save_H_real = np.savetxt('Hvalue_cs_real.txt', np.real(self.H(self.Elab_cs)), delimiter = ',')
            self.Hvalue_cs_real = torch.tensor(np.loadtxt(self.root_path + 'Hvalue_cs_real.txt'))
            save_H_imag = np.savetxt('Hvalue_cs_imag.txt', np.imag(self.H(self.Elab_cs)), delimiter = ',')
            self.Hvalue_cs_imag = torch.tensor(np.loadtxt(self.root_path + 'Hvalue_cs_imag.txt'))
            self.Hvalue_cs = self.Hvalue_cs_real + 1.j * self.Hvalue_cs_imag
            save_C0_2 = np.savetxt('C0_2value_cs.txt', self.C0_2(self.Elab_cs), delimiter = ',')
            self.C0_2value_cs = torch.tensor(np.loadtxt(self.root_path + 'C0_2value_cs.txt'))
            save_C1_2 = np.savetxt('C1_2value_cs.txt', self.C1_2(self.Elab_cs), delimiter = ',')
            self.C1_2value_cs = torch.tensor(np.loadtxt(self.root_path + 'C1_2value_cs.txt'))
            #save_cs_LO = np.savetxt('cs_LO_values.txt', self.cs_LO(), delimiter = ',')
            #self.cs_LO_values = np.loadtxt(self.root_path + 'cs_LO_values.txt')
            save_inv_cov = np.savetxt('inv_cov_matrix.txt', (np.linalg.inv(cov_matrix)).flatten(), delimiter = ',')
            self.inv_cov_matrix = torch.tensor((np.loadtxt('inv_cov_matrix.txt')).reshape((self.n, self.n)))
            #save_cov_theory = np.savetxt('cov_theory_matrix.txt', self.cov_theory().flatten(), delimiter = ',')
            #self.cov_theory_matrix = (np.loadtxt('cov_theory_matrix.txt')).reshape((self.n, self.n))
            self.Q_rest = torch.tensor(np.sum(np.log(1.0 / np.sqrt(2.0 * np.pi * np.diag(cov_matrix)))))
        else:
            # Just read in the files
            self.kvalue_cs = torch.tensor(np.loadtxt(self.root_path + 'kvalue_cs.txt'))
            self.kcvalue_cs = torch.tensor(np.loadtxt(self.root_path + 'kcvalue_cs.txt'))
            self.etavalue_cs = torch.tensor(np.loadtxt(self.root_path + 'etavalue_cs.txt'))
            self.Hvalue_cs_real = torch.tensor(np.loadtxt(self.root_path + 'Hvalue_cs_real.txt'))
            self.Hvalue_cs_imag = torch.tensor(np.loadtxt(self.root_path + 'Hvalue_cs_imag.txt'))
            self.Hvalue_cs = self.Hvalue_cs_real + 1.j * self.Hvalue_cs_imag
            self.C0_2value_cs = torch.tensor(np.loadtxt(self.root_path + 'C0_2value_cs.txt'))
            self.C1_2value_cs = torch.tensor(np.loadtxt(self.root_path + 'C1_2value_cs.txt'))
            #self.cs_LO_values = np.loadtxt(self.root_path + 'cs_LO_values.txt')
            self.inv_cov_matrix = torch.tensor((np.loadtxt('inv_cov_matrix.txt')).reshape((self.n, self.n)))
            #self.cov_theory_matrix = (np.loadtxt('cov_theory_matrix.txt')).reshape((len(self.cs_data), len(self.cs_data)))
            self.Q_rest = torch.tensor(np.sum(np.log(1.0 / np.sqrt(2.0 * np.pi * np.diag(cov_matrix)))))
            
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
        
    def __str__(self):
        return f"VI ELBO for a model with {self.param_dim} calibration parameters, {self.x_dim} dimensional input. "\
                + f"{self.nMC} MC samples are used for MC approximation."
        
    def cs_theory_bsc_in_C(self, inputs, params):
        """
        This method utilizes the existence of a bound state and also the asymptotic normalization coefficients
        to compute the theoretical cross section at the energies and angles where we have data.

        NOTE: C1+/- is really (C1+/-)^2 since we are sampling C1 squared
        
        Arguments:
        ----------
        
        inputs: torch.tensor
        Tensor of theta_cs values
        
        params: torch.tensor
        Tensor combined calibration parameters and normalizing factors [parameters, factors]  
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
    
    def cholesky_factor(self):
        """
        Computes Cholesky factor (lower triangular matrix) of a covariance function for gaussian variational family
        """
        U = torch.zeros((self.param_dim + self.f_dim ,self.param_dim + self.f_dim), dtype=torch.float64)
        U[self.up_i] = self.q_theta_c
        U[self.diag_i] = self.softplus(self.q_theta_s)
        return U.T
    
    def chi_squared(self, theory, expt, norm):
        """
         Computes chi-square value. This is an intermediate computatin for log_likelihood
        
        Arguments:
        ----------
        
        theory: torch.tensor
        Tensor of theoretical vales of an observable
        
        expt: torch.tensor
        Tensor of experimental observations
        
        norm: torch.tensor
        Tensor of normalizign factors
        """
        #print(norm.shape, theory.shape, expt.shape)
        r = norm * theory - expt
        chi_2 = r @ self.inv_cov_matrix @ r
        return chi_2, r
    
    def log_likelihood(self, inputs, exp):
        """
        Evaluates the log_likelihood for a given set of experimental inputs and experimental observations.
        
        Arguments:
        ----------
        
        inputs: torch.tensor
        Tensor of theta_cs values
        
        exp: torch.tensor
        Tensor of experimental observations
        """
        # Cast the parameters to an array
        parameters = self.params[0]

        # Unpack the erps and normalization parameters
        params = parameters[:self.erp_dim]
        params_f = parameters[self.erp_dim:]

        # If the parameters are within the bounds, compute the log likelihood
        if self.which_data == 'barnard':
            norm_barnard = params_f[-1].repeat(len(self.barnard_Elab))
            norm_som = torch.tensor([])
        if self.which_data == 'som':
            norm_barnard = torch.tensor([])
            norm_som = torch.tensor([])
            for l_int in self.l_som:
                norms = torch.cat([params_f[j].repeat(l_int[j]) for j in range(0, l_int.shape[0])])
                norm_som = torch.cat([norm_som, norms])
        if self.which_data == 'both':
            norm_barnard = params_f[-1].repeat( len(self.barnard_Elab))
            norm_som = torch.tensor([])
            for l_int in self.l_som:
                norms = torch.cat([params_f[j].repeat( l_int[j]) for j in range(0, l_int.shape[0])])
                norm_som = torch.cat([norm_som, norms])

        # Set the normalizations, theory values, and the exp values
        norm = torch.cat([norm_som, norm_barnard])
        theory = self.cs_theory_bsc_in_C(inputs,  params)
        experiment = exp

        # Compute the chi squared
        chi2 = self.chi_squared(theory, experiment, norm)[0]

        # Compute the likelihood
        log_L = -0.5 * chi2 + self.Q_rest
        return log_L
    
    def log_Normal(self, param, m, s):
        """Computes log likelihood of normal density
        
        Arguments:
        ----------
        
            param: 1 X K dimensional tensor
            m: 1 X K dimensional mean tensor
            s: 1 X K dimensional std tensor
        """
        interm__ = ((param - m) / s).pow(2)
        log_prob = - len(param) * np.log(2 * np.pi) / 2 - torch.log(s).sum() - 0.5 * interm__.sum()
        return log_prob.reshape(1,1)
        
    def q_log(self, L):
        """
        Variational family log likelihood. Gaussian with full covariance matrix
        
        Arguments:
        ----------
        
        L: torch.tensor
        Cholesky factor for scale_tril imput to torche's MultivariateNormal
        
        """
        log_lkl = torch.distributions.multivariate_normal.MultivariateNormal(self.q_theta_m.flatten(), scale_tril=L).log_prob(self.params.flatten())
        return log_lkl  
        
    def pri_log(self):
        """Prior distribution log likelohood. Gaussian"""
        log_lkl = self.log_Normal(self.params, self.prior_theta_m, self.prior_theta_s) 
        return log_lkl
    
    def generate_sample(self, n_var = 2):
        """
        Generate self.nMC from a standard multivariate normal distributino
        
        Arguments:
        ----------
        
        n_var: int
        Number of dimensions      
        """
        return Normal(0,1).sample((self.nMC,n_var,1)).to(dtype = torch.float64)
    
    def sample_reparam_normal(self, param, m, L):
                """
        Reparametrization trick.
        
        Arguments:
        ----------
        
            param: 1 X K dimensional tensor
            m: 1 X K dimensional mean tensor
            L: K X K dimensional tensor, Cholesky factor, lower triangular
        
        """
        return L @ param + m.T
    
    def compute_elbo_loop(self, x, y):
                """
        Computes MC ELBO estimate based on the log_likelihood, q_log, and pri_log methods. 
        
        Arguments:
        ----------
        
        x: torch_tensor
        Tensor of theta_cs values
        
        y: torch_tensor
        Tensor of experimental observations
        """
        
        L = self.cholesky_factor()
        z = self.generate_sample(n_var = self.param_dim + self.f_dim)
        theta = self.sample_reparam_normal(z, self.q_theta_m, L)
       

        loss = torch.randn(1,1)
        loss.data.fill_(0.0)
        
        for i in range(self.nMC):
            self.params = theta[i].T
            
            q_likelihood = self.q_log(L)
            prior_likelihood = self.pri_log()
            data_likelihood = self.log_likelihood(x, y)
            loss += data_likelihood + prior_likelihood - q_likelihood
        return loss / self.nMC
    
    def ModelString(self):
        return f"theta mean: {self.q_theta_m} \n theta std {self.softplus(self.q_theta_s)}"

def correlation_from_covariance(covariance):
    """
    Computes correlation matrix from a covariance matrix
    
    Arguments:
    ----------
    
    covariance: numpy:array
    Covariance matrix
    """
    
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation