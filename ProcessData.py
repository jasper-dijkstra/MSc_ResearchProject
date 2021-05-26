# -*- coding: utf-8 -*-
"""
Created on Wed May 26 15:44:01 2021

@author: jaspd
"""
import pandas as pd


def AnalysisDataFrame(dv, iv_list, attribute):
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
    
    df = dv.data[['NUTS_ID', 'NAME_LATN', 'N_RATIO', 'BA_RATIO']]
    
    for i, iv in enumerate(iv_list):
        if iv.ID != 3: # Population Density Dataset differs
            iv_data = iv.data[["NUTS_ID", attribute]] # Select only attribute of interest
            iv_data = iv_data.rename(columns = {"MEAN" : f"{attribute}_{iv.ID}"}) # Place correct ID at attribute
            df = pd.merge(df, iv_data, on=['NUTS_ID']) # Append to the dependent variable dataframe
        if iv.ID == 3:
            iv_data = iv.data[["NUTS_ID", "Dens2018 "]] # Select only attribute of interest
            #iv_data = iv_data.rename(columns = {"MEAN" : f"{attribute}_{iv.ID}"}) # Place correct ID at attribute
            df = pd.merge(df, iv_data, on=['NUTS_ID']) # Append to the dependent variable dataframe
    return df