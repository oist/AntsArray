#%%
import napari
import nibabel as nib
import numpy as np

img = nib.load("/home/sam-reiter/Downloads/MRI_squid brain/Squid_1.nii")
data = img.get_fdata()

viewer = napari.view_image(data, name='NIfTI Volume', colormap='gray', rendering='attenuated_mip')
napari.run()
