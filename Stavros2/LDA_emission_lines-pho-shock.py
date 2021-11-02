'''
Estimate the PCs emission lines input
'''
import astropy.coordinates as coord
import astropy.units as u
from astropy.io import ascii
from astropy.table import Table, vstack, hstack
from astropy.coordinates import SkyCoord 
import numpy as np
from pathlib import Path
import os.path
from astropy.table import Column
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy import stats
from astropy.io import fits
import argparse
import sys
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns

parser = argparse.ArgumentParser(
    description="""Make a table from the S-PLUS catalogs """)

parser.add_argument("fileName", type=str,
                    default="teste-program",
                    help="Name of table, taken the prefix ")

parser.add_argument("fileName1", type=str,
                    default="teste-program",
                    help="Name of table, taken the prefix ")

cmd_args = parser.parse_args()
file_ = cmd_args.fileName + ".txt"

cmd_args = parser.parse_args()
file_1 = cmd_args.fileName1 + ".txt"

table1 = Table.read(file_, format="ascii")
table2 = Table.read(file_1, format="ascii")

# Add a column with the label
n1 = len(table1)
label1 = np.linspace(0, 0, num=n1, dtype = int)
table1['Label'] = label1

n2 = len(table2)
label2 = np.linspace(1, 1, num=n2, dtype = int)
table2['Label'] = label2

# Merge the tables
table_merge = vstack([table1, table2])

# Put data in form expected by scikit-learn (and without col1 and col2)

X = np.array(list(zip(table_merge['col1'],
 table_merge['col2'],
 table_merge['col3'],
 table_merge['col4'],
 table_merge['col5'],
 table_merge['col6'],
 table_merge['col7'],
 table_merge['col8'],
 table_merge['col9'])))

y = table_merge["Label"]
print("Shape of array:", X.shape)

# Standarized the data
X_stand = StandardScaler().fit_transform(X)

lda = LDA(n_components=None, store_covariance=True)
lda.fit(X_stand, y)

X_lda = lda.transform(X_stand)
################################################################
print("*******************************************************************************************")
print("Porcentage:", lda.explained_variance_ratio_)
print("Porcentege sum:", sum(lda.explained_variance_ratio_))
print("Eigenvectors:", lda.scalings_)
print("Weight vector(s):", lda.coef_)
print("Covariance:", lda.covariance_)
#print("Sorted components:", lda.explained_variance_) # eigenvalues
print("*******************************************************************************************")
###############################################################

# Porcentages, eige-vectors and values
porc0 = []
ld_name = ['LD1']
porc = lda.explained_variance_ratio_ # porcantage ratio
for perc in porc:
    porc0.append(perc)

perc1 = Table([ld_name, porc0], names=('LDs', '%'), meta={'name': 'first table'})

###########################################################################################################
#eigenvectors
eginvectorss = lda.scalings_.reshape(1, 9)

einvector = Table(eginvectorss, names=('V1', 'V2', 'V3', 'V4', 'V5', 'V6',
                                          'V7', 'V8', 'V9'), meta={'name': 'first table'}) # Eigevectores

einvector["LDs"] = ld_name

new_order_eigenvetor = ['LDs', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6',
                                          'V7', 'V8', 'V9']
einvector = einvector[new_order_eigenvetor]
###########################################################################################################

data_ld = Table(X_lda, meta={'name': 'first table'})

data_ld["col0"].name = "LD1"
data_ld["Label"] = table_merge['Label']

# Mask from label
mask1 = data_ld["Label"] == 0
mask2 = data_ld["Label"] == 1
##########################
# SAVE the table results #
##########################
asciifile = "LDs_output_UV_{}.dat".format(file_1.split('_fi')[0]) 
data_ld.write(asciifile, format="ascii")

# Precentage
asciifile1 = "LDA-varience_UV_{}.dat".format(file_1.split('_fi')[0]) 
perc1.write(asciifile1, format="ascii")

# eigenvectors
asciifile3 = "LDA-eigenvectors_UV_{}.dat".format(file_1.split('_fi')[0]) 
einvector.write(asciifile3, format="ascii")

