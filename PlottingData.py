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



def FeatureImportance(ForestRegressor, labels, save_path, ForestRegressor2=None, labels2=None):
    """
    Plot the Relative Importance of each predictor variable in the Random Forest, in a bar chart
    

    Parameters
    ----------
    ForestRegressor : ensemble._forest.RandomForestRegressor
    labels : list() with names of predictor variables belonging to ForestRegressor
    ForestRegressor2 : ensemble._forest.RandomForestRegressor
    labels2 : list() with names of predictor variables belonging to ForestRegressor2
    save_path : Location (incl. extension) where output file needs to be stored.

    Returns
    -------
    Figure saved at: 'save_path'.

    """
    
    # Determine the mean Feature Importances:
    importance = ForestRegressor.feature_importances_
    
    # Determining uncertainty margins (1 std), of choosing different trees in the forest
    std = np.std([tree.feature_importances_ for tree in ForestRegressor], axis=0)

    if ForestRegressor2 == None:
        fig, ax = plt.subplots(figsize=(8.27, len(importance)/2))
        ax.barh(labels, importance, 
                xerr = std, align = 'center', alpha = 0.6, ecolor = 'black', capsize = 3)
        ax.set_xlabel("Relative Importance (%)")
    
    else:
        importance2 = ForestRegressor2.feature_importances_
        std2 = np.std([tree.feature_importances_ for tree in ForestRegressor2], axis=0)
        fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1 , 
                                       figsize = (8.27, (len(importance)+len(importance2))/2), 
                                       gridspec_kw={'height_ratios': [len(importance), len(importance2)]},
                                       sharex = True)
        
        ax1.barh(y = labels, width = importance, height = 0.8,
                 xerr = std, align = 'center', alpha = 0.6, ecolor = 'black', capsize = 3)
        ax1.text(0.975, 0.9, r"$\bf(A)$", ha='center', va='center', transform=ax1.transAxes) # Label Fig
        
        ax2.barh(y = labels2, width = importance2, height = 0.8,
                 xerr = std2, align = 'center', alpha = 0.6, ecolor = 'black', capsize = 3)
        ax2.set_xlabel("Relative Importance (%)")
        ax2.text(0.975, 0.9, r"$\bf(B)$", ha='center', va='center', transform=ax2.transAxes) # Label Fig


    #layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return



def RandomForestPerformance(rfm_1, rfm_1_estimator, rfm_2, rfm_2_estimator, save_path):
    """
    Plot the observations and predictions of the test sets of the random forest model 
    
    Parameters
    ----------
    rfm_1 : RandomForest.RandomForest
    rfm_1_estimator : ensemble._forest.RandomForestRegressor from rfm_1
    rfm_2 : RandomForest.RandomForest
    rfm_2_estimator : ensemble._forest.RandomForestRegressor from rfm_2
    save_path : Location (incl. extension) where output file needs to be stored.

    Returns
    -------
    Figure saved at: 'save_path'.

    """
    # Initiate figure
    fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize=(8.27, 3),
                                   sharex = True, sharey = True) #
    
    for ax, rfm, estimator in zip([ax1, ax2], [rfm_1, rfm_2], [rfm_1_estimator, rfm_2_estimator]):
        observed = rfm.y_test
        predicted = estimator.predict(rfm.x_test)
    
        coeff, intercept = np.polyfit(x = observed, y = predicted, deg = 1) # Determine linear regression fit
        
        ax.scatter(x = observed, y = predicted, c = 'black', marker = '+') # Plot data points as scatter
        ax.plot(observed, observed*coeff + intercept, c = 'red') # Plot regression line
        
        #ax.set_xlim(0, 1)
        #ax.set_ylim(0, 1)
        
    ax1.set_xlabel("Observed Fire Incidence \n Ratio")
    ax2.set_xlabel("Observed Burned Area \n Ratio")
    
    ax1.set_ylabel("RFM Predicted Ratio")
    
    ax1.text(0.075, 0.925, r"$\bf(A)$", ha='center', va='center', transform=ax1.transAxes)
    ax2.text(0.075, 0.925, r"$\bf(B)$", ha='center', va='center', transform=ax2.transAxes)
    
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