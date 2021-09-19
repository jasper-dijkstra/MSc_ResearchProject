# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 14:22:22 2021

@author: jaspd


This Script handles the data plots

"""


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def CorrelationMatrix(data, labels, save_path):
    """
    Plot correlation matrix as heatmap

    Parameters
    ----------
    data : pandas.DataFrame()
        Correlation Matrix.
    labels : list()
        list with label names
    save_path : str
        Location (incl. extension) where output file needs to be stored.
        
    Returns
    -------
    Figure saved at: 'save_path'.

    """
    # Create a mask so only the bottom triangle of the matrix will be displayed
    mask = np.flip(np.tri(data.shape[0], data.shape[1], k = -1))
    
    with sns.axes_style("white"):
        fig, ax = plt.subplots(figsize=(10, 10))
        #fig.tight_layout()
        
        # Plot Pearson Correlations
        fig, ax = plt.subplots(figsize=(8.27,5.845)) # Half A4
        g = sns.heatmap(data, mask = mask, # The data to be shown + maskes
                        vmin = -1, vmax = 1, cmap = 'vlag', # Identify and anchor the colormap
                        yticklabels = labels, # Set xlabels (ylabels will follow)
                        annot = True, fmt = ".2f", # Show numbers inside heatmap, up to two decimals
                        linewidths = 0.5, square=True, # technical visualisation aspects
                        center = 0, cbar_kws = {'shrink':.8}, ax=ax)
        g.set_xticklabels(labels = labels, multialignment = 'right')
        sns.set(font_scale=0.8)
        
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return



def CorrelationPlots(data, corr_idx, xitems, yitems, save_path, xlabels = None, ylabels = None):
    """
    Plot 6x2 correlation scatter plots (between variables)

    Parameters
    ----------
    data : pandas.DataFrame()
        DataFrame containing all required datapoints.
    corr_idx : pandas.DataFrame()
        DataFrame containing correlation coefficient values.
    xitems : list()
        List with names of variables to plot on x-axis. Note these names should match the ones in 'data'
        (length has to be 2!).
    yitems : list()
        List with names of variables to plot on y-axis. Note these names should match the ones in 'data'
        (length has to be 6!).
    save_path : str
        Location (incl. extension) where output file needs to be stored.
    xlabels : list(), optional
        List with label names for variables. If not specified, xitems is used as label.
    ylabels : list(), optional
        List with label names for variables. If not specified, xitems is used as label.

    Returns
    -------
    Figure saved at: 'save_path'.

    """
    # Make sure data is presented correctly for plot template
    assert len(xitems) == 2, "Variable xitems should contain exactly 2 items!"
    assert len(yitems) == 6, "Variable xitems should contain exactly 6 items!"
    
    # If no labels are provided, use the default
    if xlabels == None:
        xlabels = xitems
    if ylabels == None:
        ylabels = yitems
    
    # Subplots ID's
    alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
    
    # Initiate figure at size A4, with 6 subplots
    fig, axs = plt.subplots(nrows = 6, ncols = 2, figsize=(8.27, 11.69), dpi=320)
    
    # Plot data in each subplot
    for i, ax_row in enumerate(axs):
        for j, axes in enumerate(ax_row):
            axes.grid() # Create grid in plot
            axes.scatter(data[xitems[j]], data[yitems[i]], s=0.5) # Plot data
            if not yitems[i] == "Population Density":
                axes.set(xlabel = xlabels[j], ylabel = ylabels[i], xlim=(0,1))
            else:
                axes.set(xlabel = xlabels[j], ylabel = ylabels[i], xlim=(0,1), ylim=(0, 1000))
    
            rho = np.round(corr_idx[xitems[j]][yitems[i]], 2)
            plot_id = alphabet[(i+1)*(j+1)-1]
            axes.text(0.18, 0.9, rf"$\bf({plot_id}):$" + f' \u03C1 = {rho}', ha='center', va='center', transform=axes.transAxes)
            
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    fig.savefig(save_path, bbox_inches='tight')
    
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