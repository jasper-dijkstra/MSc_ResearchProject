# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 14:22:22 2021

@author: jaspd


This Script handles the data plots

"""


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def CorrelationMatrix(data, headers, title=None, save_path=None):
    """
    Plot correlation Matrix
    
    Inputs
    -------
    data:                       Correlation Matrix (pandas.DataFrame)
    headers:                    The x-, and y-labels to place on the fig's axes
    title (default=None):       title (str) of the figure. If none provided, no title will be added to the figure.
    save_path (default=None):   path (str) to file (incl. extension) where figure should be saved. If none provided, the figure is not saved.

    Returns
    -------
    Plotted figure and (if defined) figure saved at save_path.

    """
    
    # Create a mask so only the bottom triangle of the matrix will be displayed
    mask = np.flip(np.tri(data.shape[0], data.shape[1], k = -1))
    
    with sns.axes_style("white"):
        fig, ax = plt.subplots(figsize=(10, 10))
        #fig.tight_layout()
        
        # Plot Pearson Correlations
        ax = sns.heatmap(data, mask = mask, # The data to be shown + maskes
                         vmin = -1, vmax = 1, cmap = 'vlag', # Identify and anchor the colormap
                         xticklabels = headers[1:], yticklabels = headers[1:], # Determine labels
                         annot = True, fmt = ".2f", # Show numbers inside heatmap, up to two decimals
                         linewidths = 0.5, square=True, # technical visualisation aspects
                         center = 0, cbar_kws = {'shrink':.8})
        
        if title != None:
            ax.set_title(title)
            
        if save_path != None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return


def Scatter(x, y, fit_type = 'linear', confidence_interval = 95, order=1, title=None, save_path=None, show=False):
    

    
    # Show the figure, or not
    if not show:
        plt.ioff()
    if show:
        plt.ion()

    # Initialize a new figure
    plt.figure()
    
    # The plotting of the figure
    if not fit_type == 'log' or fit_type == 'poly': # Default is linear plot
        sns_plot = sns.regplot(x = x, y = y, marker = '+', ci = confidence_interval)
    if fit_type == 'poly':
        sns_plot = sns.regplot(x = x, y = y, marker = '+', order = 1, ci = confidence_interval)
    if fit_type == 'log': # Logarithmic fitting !!!
        sns_plot = sns.regplot(x = x, y = y, marker = '+', logistic = True, ci = confidence_interval, n_boot = 500, y_jitter=.03)
    
    if title != None:
        sns_plot.set_title(title)
        
    if save_path != None:
        fig = sns_plot.get_figure()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    
    return