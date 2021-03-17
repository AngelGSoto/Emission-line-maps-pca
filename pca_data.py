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
from astropy.table import Table

label=[]
label_dr1=[]
X = []
target = []

parser = argparse.ArgumentParser(
    description="""Make a table from the S-PLUS catalogs """)

parser.add_argument("fileName", type=str,
                    default="teste-program",
                    help="Name of table, taken the prefix ")


cmd_args = parser.parse_args()
file_ = cmd_args.fileName + ".txt"

def clean_nan_inf(M):
    mask_nan = np.sum(np.isnan(M), 1) > 0
    mask_inf = np.sum(np.isinf(M), 1) > 0
    lines_to_discard = np.logical_xor(mask_nan,  mask_inf)
    print("Number of lines to discard:", sum(lines_to_discard))
    M = M[np.logical_not(lines_to_discard), :]
    return M

    
#tab = Table.read(file_, format="ascii.no_header")
tab = np.loadtxt(file_)

XX = clean_nan_inf(tab)

#Create target to classify the kind of object
#target_ = clean_nan_inf(target_)

#XX = np.array(XX[np.logical_not(np.isnan(XX), np.isinf(XX))])
#target_ = np.array(target_[np.logical_not(np.isnan(target_), np.isinf(target_))])

print(XX.shape)

if np.any(np.isnan(XX)):
    print("NaNNNNNNNNNNNNNNNNNNNNNN")
if np.any(np.isinf(XX)):
    print("INFFFFFFFFFFFFFFFFFFFFFF")

#create the PCA for S-PLUS photometric system
XX1 = StandardScaler().fit_transform(XX)

pca = PCA(n_components=10)
pca.fit(XX1)

XX_pca = pca.transform(XX1)

#porcentages
print("Porcentage:", pca.explained_variance_ratio_)
print("Singular Value:", pca.singular_values_)
print("Component:", pca.components_[0]) # eigevectors
print("Sorted components:", pca.explained_variance_) # eigenvalues

# List with the objects

pc1, pc2, pc3 = [], [], []

#[0,1, 3,  4, 7, 10, 11, 12, 13, 8]

pc1.append(XX_pca[:, 0])
pc2.append(XX_pca[:, 1])
pc3.append(XX_pca[:, 2])

#weights

w1 = pca.components_[0]
w2 = pca.components_[1]
w3 = pca.components_[2]

print('W:',  pca.components_)
print('Ein:',  pca.explained_variance_)

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

ax1.scatter(pc1, pc2,  color= sns.xkcd_rgb["aqua"], s=130, marker='o', alpha=0.8, edgecolor='black', zorder=80.0, label='Obs. hPNe')

ax1.grid()
#lgd = ax1.legend(loc='center right', bbox_to_anchor=(1.27, 0.5), fontsize=7.5, **lgd_kws)
#ax2.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
plt.tight_layout()
#pltfile = 'Fig1-JPLUS-PC1-PC2-veri.pdf'
pltfile = 'Fig1-PC1-PC2.pdf'
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

ax2.scatter(pc1, pc3,  color= sns.xkcd_rgb["aqua"], s=130, marker='o', edgecolor='black', alpha=0.8, zorder=80.0, label='Obs. halo PNe')

ax2.legend(scatterpoints=1, ncol=2, fontsize=17.8, loc='upper center', **lgd_kws)
ax2.grid()
#ax2.grid(which='minor', lw=0.5)
#sns.despine(bottom=True)
plt.tight_layout()
#pltfile = 'Fig2-JPLUS-PC1-PC3-veri.pdf'
pltfile = 'Fig2-PC1-PC3.pdf'
file_save = os.path.join(pltfile)
plt.savefig(file_save)
plt.clf()

sys.exit()
####################################################################
#weigts ########################################################
####################################################################

filter_name = ['u', 'J0378', 'J0395', 'J0410', 'J0430', 'g', 'J0515', 'r', 'J0660', 'i', 'J0861', 'z']
color= ["#CC00FF", "#9900FF", "#6600FF", "#0000FF", "#009999", "#006600", "#DD8000", "#FF0000", "#CC0066", "#990033", "#660033", "#330034"]
#color= ["#CC0066", "#CC0066", "#CC0066", "#CC0066", "#CC0066", "#CC0066", "#CC0066", "#CC0066", "#CC0066", "#CC0066", "#CC0066", "#CC0066"]
marker = ["s", "o", "o", "o", "o", "s", "o", "s", "o", "s", "o", "s"]

filter_ = []
for a in range(1, 13):
    filter_.append(a)

plotfile = "jplus-wight1.pdf"
fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(1,1,1)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#ax.set_xlim(-10.0, 8.0)
#ax.set_ylim(-0.30, -0.28)
plt.tick_params(axis='x', labelsize = 32) 
plt.tick_params(axis='y', labelsize = 32)
ax.set_xlabel(r'Filters', size = 45)
ax.set_ylabel(r'w$_1$', size = 45)
#ax.axhline(y=0, c='k')
for wl, mag, colors, marker_ in zip(filter_, w1, color, marker):
    ax.scatter(wl, mag, color = colors, marker=marker_, s=400,  alpha=0.8,  zorder=3)
plt.xticks(filter_, filter_name, rotation=45)
#plt.xticks(filter_)
file_save = os.path.join(save_path, plotfile)    
plt.margins(0.06)
plt.subplots_adjust(bottom=0.17)
plt.tight_layout()
file_save = os.path.join(save_path, plotfile)
plt.savefig(file_save)
plt.clf()
##############################################################################
plotfile = "jplus-wight2.pdf"
fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(1,1,1)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.tick_params(axis='x', labelsize = 32) 
plt.tick_params(axis='y', labelsize = 32)
ax.set_xlabel(r'Filters', size = 45)
ax.set_ylabel(r'w$_2$', size = 45)
ax.axhline(y=0, color='k')
for wl, mag, colors, marker_ in zip(filter_, w2, color, marker):
    ax.scatter(wl, mag, color = colors, marker=marker_, s=400, alpha=0.8, zorder=3)
plt.xticks(filter_, filter_name, rotation=45)
file_save = os.path.join(save_path, plotfile)    
plt.margins(0.06)
plt.subplots_adjust(bottom=0.17)
plt.tight_layout()
file_save = os.path.join(save_path, plotfile)
plt.savefig(file_save)
plt.clf()
#############################################################################
plotfile = "jplus-wight3.pdf"
fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(1,1,1)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.set_ylim(-0.7, 0.6)
plt.tick_params(axis='x', labelsize = 32) 
plt.tick_params(axis='y', labelsize = 32)
ax.set_xlabel(r'Filters', size = 45)
ax.set_ylabel(r'w$_3$', size = 45)
ax.axhline(y=0, c='k')
for wl, mag, colors, marker_ in zip(filter_, w3, color, marker):
    ax.scatter(wl, mag, color = colors, marker=marker_, s=400, alpha=0.8, zorder=3)
plt.xticks(filter_, filter_name, rotation=45)
file_save = os.path.join(save_path, plotfile)    
plt.margins(0.06)
plt.subplots_adjust(bottom=0.17)
plt.tight_layout()
file_save = os.path.join(save_path, plotfile)
plt.savefig(file_save)
plt.clf()
