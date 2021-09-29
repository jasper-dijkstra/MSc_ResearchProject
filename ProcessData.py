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


def AnalysisDataFrame(df, iv_list, geo_name = "NUTS_ID"):
    """
    Function to generate dataframe with all attributes of interest for all independent variables
    
    Parameters
    ----------
    df : pd.DataFrame containing a 'geo_name' to append iv data to.
    iv_list : List with Independent Variable Objects
    geo_name : name of column containing geographic ID. The default is "NUTS_ID"
        use one of: "MEAN", "STD", "SUM", "Variety", "MEDIAN".

    Returns
    -------
    pd.DataFrame with values for both dependent and independent variables, to be used in regression analysis

    """
    
    header_list = df.columns.values.tolist()
    
    for i, iv in enumerate(iv_list):
        df = pd.merge(df, iv.data, on=[geo_name]) # Append to the dependent variable dataframe
        df.rename(columns = {iv.data.columns[-1] : iv.name}, inplace = True)      
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
    
    spearman_p = pd.DataFrame(float('nan'), index=df.columns[skip:], columns=df.columns[skip:])
    pearson_p = pd.DataFrame(float('nan'), index=df.columns[skip:], columns=df.columns[skip:])

    for i, x in enumerate(spearman_corr.columns):
        for j, y in enumerate(spearman_corr.columns):
            # Determine the pearson and spearman coefficients
            corrs, p_val_s = spearmanr(a = df[x], b = df[y])
            corrp, p_val_p = pearsonr(x = df[x], y = df[y])
                        
            # Append these to the DataFrame
            spearman_corr[x][y] = corrs
            pearson_corr[x][y] = corrp
            
            # Also denote the p-values
            spearman_p[x][y] = p_val_s
            pearson_p[x][y] = p_val_p
            
    return spearman_p, pearson_p, spearman_corr, pearson_corr


def PredictNUTSZone(dv, iv, y, x, estimator, geo_name = "NUTS_ID"):
    """
    Predict N_RATIO's or BA_RATIO's

    Parameters
    ----------
    dv : VariableObjects.DependentVariable
    iv : List() with VariableObjects.IndependentVariable
    y : str - Dependent Variable Attribute ("N_RATIO_Human" or "BA_RATIO_Human")
    x : list - Independent Variable Attribute(s) e.g. ["Altitude", "Population Density"]
    estimator : sklearn.ensemble._forest.RandomForestRegressor
    geo_name : str - name of the df column that describes the geographic id (e.g. NUTS_ID)
    
    Returns
    -------
    y_hat : The predictions from the RFM, per geo_name

    """

    df_predict, headers_ = AnalysisDataFrame(dv.data_with_nan, iv) # Initiate DataFrame to be used for the analysis

    filtered = df_predict[df_predict[y].isna()] # Filter to the rows containing 'nan's
    filtered = filtered[[geo_name] + list(x.columns)] # Filter all columns of interest
    
    y_hat = pd.DataFrame(filtered[geo_name]) # Initiate df to add predictions to
    y_hat[f"Exp_{y}"] = estimator.predict(filtered.iloc[:,1:]) # Now Predict, using the RFM estimator
    
    return y_hat

