#!/usr/bin/env/python3
# -*- coding: utf-8 -*-
import sys
import numpy as np



class DataLoader(object):
    """
    This class loads and selects the subset of the data we want to work with.
    It returns the data in the 'get_data()' method.
    It also handles the normalization information in the 'get_normalization_info()' method.

    E_min : [0.676, 0.84 , 1.269, 1.741, 2.12 , 2.609, 2.609, 3.586, 4.332, 5.475]
    E_max : [0.706, 0.868, 1.292, 1.759, 2.137, 2.624, 2.624, 3.598, 4.342, 5.484]
    """
    def __init__(self, E_min : float, E_max : float, which_data : str):
        # Store into self
        self.E_min = E_min
        self.E_max = E_max
        self.which_data = which_data
        
        # Load the data
        self.all_data = np.load('./data/full_data.npy')

        # Select the data larger than E_min and smaller than E_max
        self.selected_data = self.all_data[np.where(np.logical_and((self.all_data[:, 0] >= self.E_min), (self.all_data[:, 0] <= self.E_max)))]

        # Now we select the data set we want
        if self.which_data == 'both':
            self.data = self.selected_data
        elif self.which_data == 'som':
            self.data = self.selected_data[np.where(self.selected_data[:, 4] < 10)]
        elif self.which_data == 'barnard':
            self.data = self.selected_data[np.where(self.selected_data[:, 4] == 10)]
        else:
            sys.exit('Data set not recognized. Please select either "both", "som", or "barnard".')



    def get_data(self):
        """
        Returns the data stored in the DataLoader object.

        Returns:
            The data stored in the DataLoader object.
        """
        return self.data



    def get_normalization_priors(self):
        """
        Get the normalization priors for the data.

        This method calculates the normalization priors for the data based on the selected data set.
        The priors are Gaussian distributions centered at 1.0 with different standard deviations (sigmas).
        The means (mus) are set to 1.0 for all sigmas.
        The bounds for the priors are set to [0, 2] for all sigmas.
        The normalization prior information is returned as a numpy array.

        Returns:
            numpy.ndarray: The normalization prior information, which includes the bounds, means, and sigmas.
        """
        # All the priors are gaussian centered at 1.0 with different sigmas
        som_f_sigmas = np.array([0.064, 0.076, 0.098, 0.057, 0.045, 0.062, 0.041, 0.077, 0.063, 0.089])
        barnard_f_sigma = np.array([0.05])

        # Select the sigmas
        if self.which_data == 'som':
            sigmas = som_f_sigmas[np.unique(self.data[:, 4].astype(int))]
        elif self.which_data == 'barnard':
            sigmas = barnard_f_sigma
        elif self.which_data == 'both':
            sigmas = np.concatenate([som_f_sigmas[np.unique(self.data[:, 4].astype(int))], barnard_f_sigma])
        else:
            sys.exit('Data set not recognized. Please select either "both", "som", or "barnard".')
        
        # Specify the means
        mus = np.ones(sigmas.shape)

        # Get the means and sigmas together + generate the bounds
        gauss_prior_fs = np.vstack([mus, sigmas]).T
        bounds = np.array([[0, 2] for i in sigmas])

        # Combine all together
        norm_prior_info = np.hstack([bounds, gauss_prior_fs])
        return norm_prior_info