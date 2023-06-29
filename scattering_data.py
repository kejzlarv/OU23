#!/usr/bin/env/python3
# -*- coding: utf-8 -*-
import sys
import numpy as np


# Data Loader 
class DataLoader:
    def __init__(self, E_max, which_data):
        """
        This class will read in 3He-alpha scattering data from the Som and Barnard
        datasets.

        Arguments:
        ----------
        E_max : float
        Determines the maximum energy of the data to consider.

        which_data : string
        Determines which data to look at
        """
        
        ##############################################################################
                    # Set the initial constants and useful variables
        ##############################################################################
        # Set the beam and target parameters
        self.Zb = 2 # Charge number
        self.Zt = 2 # Charge number
        self.mb = 2809.43 # MeV - 3He
        self.mt = 3728.42 # MeV - alpha
        self.mu = (self.mb * self.mt) / (self.mb + self.mt) # Reduced mass

        # Set useful constants and variables
        self.alpha = 1 / 137.036 # dimless (fine structure alpha_em)
        self.h_bar_c = 197.327 # MeV fm
        self.root_path = './'

        # Set which data and method we are going to use
        self.E_max = E_max
        self.which_data = which_data

        ##############################################################################
                                # Read in and set the data #
        ##############################################################################
        # Set up the list of paths to the data
        barnard_paths = [self.root_path + 'barnard_data/barnard5477.txt',
                        self.root_path + 'barnard_data/barnard6345.txt',
                        self.root_path + 'barnard_data/barnard7395.txt',
                        self.root_path + 'barnard_data/barnard9003.txt',
                        self.root_path + 'barnard_data/barnard10460.txt',
                        self.root_path + 'barnard_data/barnard11660.txt',
                        self.root_path + 'barnard_data/barnard12530.txt',
                        self.root_path + 'barnard_data/barnard14080.txt']

        som_paths = [self.root_path + 'SOM/som_cm_int1.npy',
                    self.root_path + 'SOM/som_cm_int2.npy',
                    self.root_path + 'SOM/som_cm_int3.npy']

        # Handle Barnard set first
        barnard_data_list = []
        for path in barnard_paths:
            barnard_data_list.append(np.loadtxt(path))

        # Handle Som set
        som_data_list = []
        l_som = [] # For Som normalization
        l_som_energies = []
        for path in som_paths:
            data_set = np.load(path, allow_pickle = True)
            som_data_list.append(data_set)
            l_som_energies.append([data_set[i][:, 0][0] for i in range(len(data_set))])
            l_som.append([len(data_set[j]) for j in range(len(data_set))])

        # Convert the lists into arrays
        barnard_data = np.concatenate(barnard_data_list)
        som_data = np.array(som_data_list)
        self.l_som_energies = np.array(l_som_energies)
        self.l_som = np.array(l_som)

        # Set up variables for bounds on parameters (widths of prior)
        som_f_sigmas = np.array([0.064, 0.076, 0.098, 0.057, 0.045, 0.062,
                            0.041, 0.077, 0.063, 0.089])
        barnard_f_sigma = np.array([0.05])

        # Note: *** The Barnard data has lowest E at 2.439 MeV and the 
        # Som data has lowest E at 0.676 MeV ***
        # Truncate the data based on the energy cutoff
        if E_max != None:
            if E_max < 0.706:
                sys.stderr.write('No data exists with E < 0.706!')
                sys.exit(-1)
            else:
                # Barnard first
                # Pulls the indices where the energy is less than E_max
                barnard_indices = np.where(barnard_data[:, 1] <= E_max)
                # Only pulls the correct indices
                barnard_data = barnard_data[barnard_indices]
                # Som next
                # Pulls the indices where the energy is less than E_max for each interaction region
                som_indices = np.array([np.where(int_region_energy <= E_max) for int_region_energy in self.l_som_energies])
                # Selects the correct indices from before and concatenates them
                som_data = np.concatenate([som_data_list[i][som_indices[i]] for i in range(0, len(l_som))])
                # Pulls the correct number of data points for each energy
                self.l_som = np.array(np.concatenate([self.l_som[i][som_indices[i]] for i in range(0, len(self.l_som))]))
                # Select all the som_f_sigmas we need 
                som_f_sigmas = som_f_sigmas[:self.l_som.shape[1]]

        # Convert Som data to an array
        som_data = np.concatenate(np.concatenate(som_data))

        # Som has the data formated by [E, theta, cs, err] but Barnard has [theta, E, cs, err]
        # so I need to standardize this. I will match the Barnard convention
        temp_som = som_data.copy()
        som_data = np.column_stack([temp_som[:, 1], temp_som[:, 0], temp_som[:, 2], temp_som[:, 3]])

        # Now select which data to use
        if self.which_data == 'both':
            self.data = np.concatenate([barnard_data, som_data])
            self.f_sigmas = np.concatenate([som_f_sigmas, barnard_f_sigma])
            self.barnard_Elab = barnard_data[:, 1] # For normalization purposes
        elif self.which_data == 'som':
            self.data = som_data
            self.f_sigmas = np.concatenate([som_f_sigmas])
        elif self.which_data == 'barnard':
            self.data = barnard_data
            self.f_sigmas = np.concatenate([barnard_f_sigma])
            self.barnard_Elab = barnard_data[:, 1] # For normalization purposes
        else:
            sys.stderr.write('Choose a \'which_data\': both, som, barnard...')
            sys.exit(-1)

        # We can unpackage the data into it's different components
        self.theta_cs = self.data[:, 0]
        self.Elab_cs = self.data[:, 1]
        self.cs_data = self.data[:, 2] # these are experimental observations (the y's)
        self.err_cs = self.data[:, 3] # these are standard deviations to be used to construct likelihood matrix
        
        self.f_mus = np.ones(self.f_sigmas.shape[0])
        self.f_bounds = np.array([[0, 2] for i in self.f_sigmas])