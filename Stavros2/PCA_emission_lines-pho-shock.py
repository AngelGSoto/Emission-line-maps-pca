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
from sklearn.decomposition import PCA
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

print("Shape of array:", X.shape)

# Standarized the data
X_stand = StandardScaler().fit_transform(X)

# Creating the PCA 
pca = PCA(n_components=5)
pca.fit(X_stand)

X_pca = pca.transform(X_stand)
################################################################
print("*******************************************************************************************")
print("Porcentage:", pca.explained_variance_ratio_)
print("Porcentege sum:", sum(pca.explained_variance_ratio_))
print("Singular Value:", pca.singular_values_)
print("Component:", pca.components_) # eigevectors
print("Sorted components:", pca.explained_variance_) # eigenvalues
print("*******************************************************************************************")
###############################################################
# Porcentages, eige-vectors and values
porc0 = []
pc_name = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']
porc = pca.explained_variance_ratio_ # porcantage ratio
for perc in porc:
    porc0.append(perc)

perc1 = Table([pc_name, porc0], names=('PCs', '%'), meta={'name': 'first table'})

###########################################################################################################
einvector = Table(pca.components_, names=('V1', 'V2', 'V3', 'V4', 'V5', 'V6',
                                          'V7', 'V8', 'V9'), meta={'name': 'first table'}) # Eigevectores

einvector["PCs"] = pc_name

new_order_eigenvetor = ['PCs', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6',
                                          'V7', 'V8', 'V9']

einvector = einvector[new_order_eigenvetor]

############################################################################################################
einvalue1 = []
einvalue = pca.explained_variance_ # Eigenvalues
for einvalue0 in einvalue:
    einvalue1.append(einvalue0)

einvalue2 = Table([pc_name, einvalue1], names=('PCs', 'EigenValues'), meta={'name': 'first table'})
###########################################################################################################

data_pc = Table(X_pca, names=('PC1', 'PC2', 'PC3', 'PC4', 'PC5'), meta={'name': 'first table'})

data_pc["Label"] = table_merge['Label']

# Mask from label
mask1 = data_pc["Label"] == 0
mask2 = data_pc["Label"] == 1

##########################
# SAVE the table results #
##########################
asciifile = "PCs_output_UV_{}.dat".format(file_1.split('_fi')[0]) 
data_pc.write(asciifile, format="ascii")

# Precentage
asciifile1 = "varience_UV_{}.dat".format(file_1.split('_fi')[0]) 
perc1.write(asciifile1, format="ascii")

# eigenvalues
asciifile2 = "eigenvalues_UV_{}.dat".format(file_1.split('_fi')[0]) 
einvalue2.write(asciifile2, format="ascii")

# eigenvectors
asciifile3 = "eigenvectors_UV_{}.dat".format(file_1.split('_fi')[0]) 
einvector.write(asciifile3, format="ascii")

#################################
# Some plots            #########
#################################
lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': None}
#sns.set(style="dark")#, context="talk")
#sns.set_style('ticks')       
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(111)
# ax1.set_xlim(-8.2, 5.7)
# ax1.set_ylim(-2.5, 1.5)
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# ax1.set_xlim(-10, 10)
# ax1.set_ylim(-3.3, 3.2)
#ax1.set_xlim(xmin=-2.5,xmax=2.0)
plt.tick_params(axis='x', labelsize=32) 
plt.tick_params(axis='y', labelsize=32)
plt.xlabel(r'PC1', fontsize= 35)
plt.ylabel(r'PC2', fontsize= 35)
ax1.scatter(data_pc["PC1"][mask1], data_pc["PC2"][mask1],  color= sns.xkcd_rgb["aqua"], s=130, marker='o', alpha=0.8, edgecolor='black', zorder=80.0, label='UV')
ax1.scatter(data_pc["PC1"][mask2], data_pc["PC2"][mask2],  color= sns.xkcd_rgb["pale yellow"], s=130, marker='o', alpha=0.8, edgecolor='black', zorder=80.0, label='Shock Allen')

ax1.grid()
ax1.legend(scatterpoints=1, ncol=1, fontsize=17.8, loc='upper left', **lgd_kws)
#ax2.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
plt.tight_layout()
#pltfile = 'Fig1-JPLUS-PC1-PC2-veri.pdf'
pltfile = 'Fig1-PC1-PC2_UV_{}.jpg'.format(file_1.split('_fi')[0])
save_path = ' '
file_save = os.path.join(pltfile)
plt.savefig(file_save)
plt.clf()

####################################################################
#PC1 vs PC3 ########################################################
####################################################################

lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': None}
#sns.set(style="dark")#, context="talk")
#sns.set_style('ticks')       
fig = plt.figure(figsize=(12, 8))
ax2 = fig.add_subplot(111)
# ax2.set_xlim(-10.0, 8.0)
# ax2.set_ylim(-2.0, 1.5)
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# ax2.set_xlim(-8.2, 5.7)
# ax2.set_ylim(-1.1, 1.0)
#ax1.set_xlim(xmin=-2.5,xmax=2.0)
plt.tick_params(axis='x', labelsize=32) 
plt.tick_params(axis='y', labelsize=32)
plt.xlabel(r'PC1', fontsize= 35)
plt.ylabel(r'PC3', fontsize= 35)

ax2.scatter(data_pc["PC1"][mask1], data_pc["PC3"][mask1],  color= sns.xkcd_rgb["aqua"], s=130, marker='o', alpha=0.8, edgecolor='black', zorder=80.0, label='UV')
ax2.scatter(data_pc["PC1"][mask2], data_pc["PC3"][mask2],  color= sns.xkcd_rgb["pale yellow"], s=130, marker='o', alpha=0.8, edgecolor='black', zorder=80.0, label='Shock Allen')

ax2.legend(scatterpoints=1, ncol=1, fontsize=17.8, loc='upper left', **lgd_kws)
ax2.grid()
#ax2.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
#pltfile = 'Fig2-JPLUS-PC1-PC3-veri.pdf'
pltfile1 = 'Fig2-PC1-PC3_UV_{}.jpg'.format(file_1.split('_fi')[0])
file_save1 = os.path.join(pltfile1)
plt.savefig(file_save1)
plt.clf()

####################################################################
#PC1 vs PC3 ########################################################
####################################################################

lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': None}
#sns.set(style="dark")#, context="talk")
#sns.set_style('ticks')       
fig = plt.figure(figsize=(12, 8))
ax2 = fig.add_subplot(111)
# ax2.set_xlim(-10.0, 8.0)
# ax2.set_ylim(-2.0, 1.5)
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# ax2.set_xlim(-8.2, 5.7)
# ax2.set_ylim(-1.1, 1.0)
#ax1.set_xlim(xmin=-2.5,xmax=2.0)
plt.tick_params(axis='x', labelsize=32) 
plt.tick_params(axis='y', labelsize=32)
plt.xlabel(r'PC2', fontsize= 35)
plt.ylabel(r'PC3', fontsize= 35)

ax2.scatter(data_pc["PC2"][mask1], data_pc["PC3"][mask1],  color= sns.xkcd_rgb["aqua"], s=130, marker='o', alpha=0.8, edgecolor='black', zorder=80.0, label='UV')
ax2.scatter(data_pc["PC2"][mask2], data_pc["PC3"][mask2],  color= sns.xkcd_rgb["pale yellow"], s=130, marker='o', alpha=0.8, edgecolor='black', zorder=80.0, label='Shock Allen')

ax2.legend(scatterpoints=1, ncol=2, fontsize=17.8, loc='upper center', **lgd_kws)
ax2.grid()
#ax2.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
#pltfile = 'Fig2-JPLUS-PC1-PC3-veri.pdf'
pltfile1 = 'Fig3-PC2-PC3_UV_{}.jpg'.format(file_1.split('_fi')[0])
file_save1 = os.path.join(pltfile1)
plt.savefig(file_save1)
plt.clf()
