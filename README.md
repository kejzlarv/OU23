# Bayesian uncertainty quantification with Pytorch (estimation of 3.He-alpha scattering parameters)

Bayesian estimation of 3.He-alpha scattering parameters (PyTorch implementation). Both the mean-field approximation and the linear normalizing flow approximation are implemented.

## Content of the repository:
1. [VBI_scattering.ipynb](VBI_scattering.ipynb) - Notebook that contains the complete code for the VBI calibration of scattering data. This notebook is meant as a playground. For complete and polished code, see Model_fitting_and_analysis notebook, vbi_elbo.pi, and scattering_data.py
2. [Model_fiting_and_analysis.ipynb](Model_fiting_and_analysis.ipynb) - Notebook with the VBI analyisis of the scattering data. This is the current frontend.
3. [vbi_elbo.py](vbi_elbo.py) - Torch-based implementation of ELBO for mean-field VBI and VBI with full Gaussian variational family. It is imported in Model_fiting_and_analysis.ipynb.
4. [scattering_data.py](scattering_data.py) - DataLoader for the scattering data. It is imported in Model_fiting_and_analysis.ipynb.
