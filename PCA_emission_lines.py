'''
Estimate the PCs emission lines input
'''
import astropy.coordinates as coord
import astropy.units as u
from astropy.io import ascii
from astropy.table import Table
from astropy.coordinates import SkyCoord 
import numpy as np
from pathlib import Path
from astropy.table import Column
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy import stats
from astropy.io import fits
import argparse
import sys

parser = argparse.ArgumentParser(
    description="""Make a table from the S-PLUS catalogs """)

parser.add_argument("fileName", type=str,
                    default="teste-program",
                    help="Name of table, taken the prefix ")

cmd_args = parser.parse_args()
file_ = cmd_args.fileName + ".txt"

table = Table.read(file_, format="ascii")

# Put data in form expected by scikit-learn (and without col1 and col2)
table.remove_columns(['col1', 'col2'])

X = np.array(list(zip(table['col3'],
 table['col4'],
 table['col5'],
 table['col6'],
 table['col7'],
 table['col8'],
 table['col9'],
 table['col10'],
 table['col11'],
 table['col12'],
 table['col13'],
 table['col14'],
 table['col15'],
 table['col16'],
 table['col17'],
 table['col18'],
 table['col19'],
 table['col20'],
 table['col21'])))

print("Shape of array:", X.shape)

# Standarized the data
X_stand = StandardScaler().fit_transform(X)

# Creating the PCA 
pca = PCA(n_components=10)
pca.fit(X_stand)

X_pca = pca.transform(X_stand)

# Porcentages, eige-vectors and values
porc0 = []
pc_name = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6',
                               'PC7', 'PC8', 'PC9', 'PC10']
porc = pca.explained_variance_ratio_ # porcantage ratio
for perc in porc:
    porc0.append(perc)

perc1 = Table([pc_name, porc0], names=('PCs', '%'), meta={'name': 'first table'})

###########################################################################################################
einvector = Table(pca.components_, names=('V1', 'V2', 'V3', 'V4', 'V5', 'V6',
                                          'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16',
                                          'V17', 'V18', 'V19'), meta={'name': 'first table'}) # Eigevectores

einvector["PCs"] = pc_name

new_order_eigenvetor = ['PCs', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6',
                                          'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16',
                                          'V17', 'V18', 'V19']

einvector = einvector[new_order_eigenvetor]

############################################################################################################
einvalue1 = []
einvalue = pca.explained_variance_ # Eigivalues
for einvalue0 in einvalue:
    einvalue1.append(einvalue0)

einvalue2 = Table([pc_name, einvalue1], names=('PCs', 'EigenValues'), meta={'name': 'first table'})
###########################################################################################################


print("Porcentage:", pca.explained_variance_ratio_)
print("Singular Value:", pca.singular_values_)
print("Component:", pca.components_) # eigevectors
print("Sorted components:", pca.explained_variance_) # eigenvalues

# Reshape the final PCs
pc1 = X_pca[:, 0].reshape((250, 310))
pc2 = X_pca[:, 1].reshape((250, 310))
pc3 = X_pca[:, 2].reshape((250, 310))
pc4 = X_pca[:, 3].reshape((250, 310))
pc5 = X_pca[:, 4].reshape((250, 310))
pc6 = X_pca[:, 5].reshape((250, 310))
pc7 = X_pca[:, 6].reshape((250, 310))
pc8 = X_pca[:, 7].reshape((250, 310))
pc9 = X_pca[:, 8].reshape((250, 310))
pc10 = X_pca[:, 9].reshape((250, 310))

# Definition to write the FITS file with each PC
def FitsFile(pc, label, number):
    hdu = fits.PrimaryHDU(pc)
    hdu.header
    hdu.header['PCA dimension'] = label
    hdul = fits.HDUList([hdu])
    hdul.writeto('PC{}_2D_{}.fits'.format(number, file_.split('ut_')[-1].split('.tx')[0]), overwrite=True)

# Save the PCs resulting
#####
# FITS file
FitsFile(pc1, "PC1", 1)
FitsFile(pc2, "PC2", 2)
FitsFile(pc3, "PC3", 3)
FitsFile(pc4, "PC4", 4)
FitsFile(pc5, "PC5", 5)
FitsFile(pc6, "PC6", 6)
FitsFile(pc7, "PC7", 7)
FitsFile(pc8, "PC8", 8)
FitsFile(pc9, "PC9", 9)
FitsFile(pc10, "PC10", 10)

# ASCII file
data = Table(X_pca, names=('PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6',
                               'PC7', 'PC8', 'PC9', 'PC10'), meta={'name': 'first table'})
# Add the pixels
x, y = [], []
for i in range(len(pc1[:,0])):
    for j in range(len(pc1[0])):
        x.append(i)
        y.append(j)       
data['x'] = x
data['y'] = y

#reorganizing the colunm names
new_order = ['x', 'y', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10']
data_new = data[new_order]

##########################
# SAVE the table results #
##########################
asciifile = "PCs_output_{}.dat".format(file_.split('ut_')[-1].split('.tx')[0]) 
data_new.write(asciifile, format="ascii")

# Precentage
asciifile1 = "varience_{}.dat".format(file_.split('ut_')[-1].split('.tx')[0]) 
perc1.write(asciifile1, format="ascii")

# eigenvalues
asciifile2 = "eigenvalues_{}.dat".format(file_.split('ut_')[-1].split('.tx')[0]) 
einvalue2.write(asciifile2, format="ascii")

# eigenvectors
asciifile3 = "eigenvectors_{}.dat".format(file_.split('ut_')[-1].split('.tx')[0]) 
einvector.write(asciifile3, format="ascii")
