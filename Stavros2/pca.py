'''
Principal component analysis (J-PLUS)
'''
#from __future__ import print_function
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy import stats
import sys
import glob
import json
import seaborn as sns
import os.path
from collections import OrderedDict
from scipy.stats import gaussian_kde
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from scipy.optimize import fsolve
import argparse
from pathlib import Path
from astropy.table import Table, vstack, hstack
import sys

label=[]
label_dr1=[]
X = []
target = []

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
 table_merge['col9'],
 table_merge['Label'])))

print("Shape of array:", X.shape)

# Standarized the data
X_stand = StandardScaler().fit_transform(X)

# Creating the PCA 
pca = PCA(n_components=9)
pca.fit(X_stand)

X_pca = pca.transform(X_stand)

# Add PCs to table
pca_table = Table(X_pca, names=('PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6',
                               'PC7', 'PC8', 'PC9'), meta={'name': 'first table'})
final_table = hstack([table_merge, pca_table])

mask1 =  final_table['Label'] == 0
mask2 =  final_table['Label'] == 1

final_table1 = final_table[mask1]
final_table2 = final_table[mask2]


# Porcentages and others parameters
print("Porcentage:", pca.explained_variance_ratio_)
print("Singular Value:", pca.singular_values_)
print("Component:", pca.components_[0]) # eigevectors
print("Sorted components:", pca.explained_variance_) # eigenvalues

fig, ax = plt.subplots(figsize=(15, 15)), plt.axes(projection='3d')

ax.scatter3D(final_table1["PC1"], final_table1["PC2"], final_table1["PC3"],
             c=sns.xkcd_rgb['aqua'], s=100, marker="o", edgecolor='black', alpha=0.7,
            cmap=plt.cm.get_cmap('Accent', 10))
ax.scatter3D(final_table2["PC1"], final_table2["PC2"], final_table2["PC3"],
            c=sns.xkcd_rgb['dark pink'], s=100,  marker="o", edgecolor='black', alpha=0.7,
            cmap=plt.cm.get_cmap('Accent', 10))
ax.set_xlabel('component 1')
ax.set_ylabel('component 2')
ax.set_zlabel('component 3')
#ax.set(xlim=[-5.0, 20.0], ylim=[-5.0, 10.0], zlim=[-5.0, 6.0])

#ax.set_aspect("equal")
plt.savefig("PC3D_uv_shocks.pdf")

# lgd_kws = {'frameon': True, 'fancybox': True, 'shadow': None}
# #sns.set(style="dark")#, context="talk")
# #sns.set_style('ticks')       
# fig = plt.figure(figsize=(12, 8))
# ax1 = fig.add_subplot(111)
# # ax1.set_xlim(-8.2, 5.7)
# # ax1.set_ylim(-2.5, 1.5)
# ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# # ax1.set_xlim(-10, 10)
# # ax1.set_ylim(-3.3, 3.2)
# #ax1.set_xlim(xmin=-2.5,xmax=2.0)
# plt.tick_params(axis='x', labelsize=32) 
# plt.tick_params(axis='y', labelsize=32)
# plt.xlabel(r'PC1', fontsize= 35)
# plt.ylabel(r'PC2', fontsize= 35)

# ax1.scatter(final_table1["PC1"], final_table1["PC2"], color= sns.xkcd_rgb["aqua"], s=130, marker='o', alpha=0.8, edgecolor='black', zorder=80.0, label='UV')
# ax1.scatter(final_table2["PC1"], final_table2["PC2"], color= sns.xkcd_rgb["green"], s=130, marker='o', alpha=0.8, edgecolor='black', zorder=80.0, label='Allen')

# ax1.grid()
# #lgd = ax1.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)
# #ax2.grid(which='minor', lw=0.5)
# #sns.despine(bottom=True)
# plt.tight_layout()
# plt.tight_layout()
# #pltfile = 'Fig1-JPLUS-PC1-PC2-veri.pdf'
# pltfile = 'Fig1-PC1-PC2.pdf'
# save_path = ' '
# file_save = os.path.join(pltfile)
# plt.savefig(file_save)
# plt.clf()

