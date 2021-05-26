# -*- coding: utf-8 -*-
"""
Created on Wed May 19 10:41:14 2021

author: Jasper Dijkstra

This script contains functions that read input datasets and return as pd.df or np.array

For example, data specific:
    - Population Density: https://ec.europa.eu/eurostat/web/products-datasets/-/demo_r_d3dens

Or general function:
    - ReadNC: Reads an input NetCDF file and returns lat, lon and data.


"""

import numpy as np
import pandas as pd
import netCDF4 as nc


def ReadPopulationDensityTSV(tsv_path, dependent_variable_obj, years = 2018):
    """
    
    Parameters
    ----------
    tsv_path : str
        Path to the tsv file, which can be found via: 
            https://ec.europa.eu/eurostat/web/products-datasets/-/demo_r_d3dens
    dependent_variable_obj : ClassObjects.DependentVariable
        Object generated in Dependent Variable Class.
    years : int, list, optional
        The year(s) for which data has to be returned. This can be either an integer or a list with integers.
        The default is 2018, since this is the most recent year that has data for all desired NUTS3 areas.

    Returns
    -------
    df : pandas.core.frame.DataFrame
        DataFrame with columns: "UNIT", "NUTS_ID", "NUTS_Area", *columns with population density and total 
        population for all specified input years.

    """
    # Get the years variable in desired format:
    assert isinstance(years, int) or isinstance(years, list), \
        "The 'years' parameter should be a list or int, not {}.".format(type(years))
    if isinstance(years, int):
        years = [f"{years} "]
    else:
        inputyears = years.copy()
        years = []
        for year in inputyears:
            years.append(f"{year} ")
    
    # Open the tsv file
    df = pd.read_csv(tsv_path, delimiter="\t")
    
    # Split the units from the NUTS code and remove the original
    first_cols = df.iloc[:,0].str.split(",", expand = True) 
    df = pd.merge(first_cols, df, left_index = True, right_index=True)
    df.rename(columns={0:"UNIT", 1:"NUTS_ID"}, inplace=True, errors="raise")
    df = df.drop(df.columns[2], axis = 1)
    
    # Filter to the required NUTS regions
    NUTS_Regions = list(dependent_variable_obj.data["NUTS_ID"]) # List required NUTS regions
    df = df[df["NUTS_ID"].isin(NUTS_Regions)] # Remove those that do not occur in the list
    
    # Filter to the required years
    df = pd.concat([df[df.columns[:2]], df[years]], axis = 1)
    
    # Add a NUTS area component to the df
    area_df = dependent_variable_obj.data[["NUTS_ID", "NUTS_Area"]]
    df = pd.merge(df, area_df, on=['NUTS_ID'])
    
    # Apply the following steps for each year column in the df:
    for year in years:
        # Remove all non-number characters (except ".")
        df[year] = df[year].str.replace(r"[^.0-9\s]", '')
        
        # Convert all str population density values to a number
        df[year] = df[year].apply(pd.to_numeric, errors = 'coerce')
        
        # Get the total population
        df.insert(loc = df.columns.get_loc(year)+1, column = f"Total{year}", value = df[year]*df["NUTS_Area"])
        
        # Rename Year Column, for more clarity
        df.rename(columns = {year : f"Dens{year}"}, inplace = True)
        
    # All columns to numbers, isntead of string
    #df[df.columns[2:]] = df[df.columns[2:]].apply(pd.to_numeric, errors = 'coerce')
    return df


def ReadNC(path, variable):
    """
    General functino to read NetCDF file
    
    Parameters
    ----------
    path : str
        Path to the nc file.
    variable : str
        The name of the variable in the file that contains the data values of interest.

    Returns
    -------
    lon : np.array
        1D Array with all longitude coordinates of the grid cells.
    lat : np.array
        1D Array with all latitude coordinates of the grid cells..
    data : np.array
        2D array of size [lat, lon] that describes the value of the grid cells.

    """
    # Open dataset
    fid = nc.Dataset(path)
    
    # Get Coordinates
    lon = np.ma.filled(fid.variables['Longitude'][:], np.nan)
    lat = np.ma.filled(fid.variables['Latitude'][:], np.nan)
    
    # Get Dataset of interest
    data = np.ma.filled(fid.variables[variable][:], np.nan)
    
    # Make the data the correct way up
    data = np.flipud(data)
    
    # Close the dataset
    fid.close()
    
    return lon, lat, data

