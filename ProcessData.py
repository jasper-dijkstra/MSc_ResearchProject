# -*- coding: utf-8 -*-
"""
Created on Wed May 26 15:44:01 2021

@author: jaspd

Script with functions to process input input data:
    - AnalysisDataFrame: Generates DataFrame that facilitates later analysis
    - CorrMatrix: Generates a pearson and spearman correlation matrix from input AnalysisDataFrame
    
"""

import pandas as pd
from scipy.stats import spearmanr, pearsonr


def AnalysisDataFrame(dv, iv_list):
    """
    Function to generate dataframe with all attributes of interest for all independent variables
    
    Parameters
    ----------
    dv : Dependent Variable Object.
    iv_list : List with Independent Variable Objects
    attribute : Attributes from iv's to use to explain dependent variable 
        use one of: "MEAN", "STD", "SUM", "Variety", "MEDIAN".

    Returns
    -------
    pd.DataFrame with values for both dependent and independent variables, to be used in regression analysis

    """
    
    df = dv.data[['NUTS_ID'] + dv.data_header[-4:]]
    header_list = df.columns.values.tolist()
    
    for i, iv in enumerate(iv_list):
        #iv_data = iv.data.head()[-1]
        #iv_data = iv.data[["NUTS_ID", attribute]] # Select only attribute of interest
        #iv_data = iv_data.rename(columns = {"MEAN" : f"{attribute}_{iv.ID}"}) # Place correct ID at attribute
        df = pd.merge(df, iv.data, on=['NUTS_ID']) # Append to the dependent variable dataframe
        header_list.append(iv.name)
        
    return df, header_list


def CorrMatrix(df, skip=0):
    """
    Function that creates two correlation matrices with spearman and pearson correlations from input df

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with variables to find correlations of.
    skip : int
        Amount of left columns of the df that will be skipped for the correlation. The default is 0.
        
    Returns
    -------
    spearman_corr : pd.DataFrame
        Correlation Matrix with spearman correlations.
    pearson_corr : pd.DataFrame
        Correlation Matrix with pearson correlations.

    """
    spearman_corr = pd.DataFrame(float('nan'), index=df.columns[skip:], columns=df.columns[skip:])
    pearson_corr = pd.DataFrame(float('nan'), index=df.columns[skip:], columns=df.columns[skip:])
    
    for i, x in enumerate(spearman_corr.columns):
        for j, y in enumerate(spearman_corr.columns):
            # Determine the pearson and spearman coefficients
            corrs, p_val_s = spearmanr(a = df[x], b = df[y])
            corrp, p_val_p = pearsonr(x = df[x], y = df[y])
            
            # Append these to the DataFrame
            spearman_corr[x][y] = corrs
            pearson_corr[x][y] = corrp
    
    return spearman_corr, pearson_corr