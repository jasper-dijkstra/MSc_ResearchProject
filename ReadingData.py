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


def ReadPopulationDensityTSV(tsv_path, dependent_variable_obj, years = list(np.arange(2001, 2019, 1))):
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
        df[year] = df[year].str.replace(r"[^.0-9\s]", "", regex = True)
        
        # Convert all str population density values to a number
        df[year] = df[year].apply(pd.to_numeric, errors = 'coerce')
        
        # Get the total population
        df.insert(loc = df.columns.get_loc(year)+1, column = f"Total{year}", value = df[year]*df["NUTS_Area"])
        
        # Rename Year Column, for more clarity
        df.rename(columns = {year : f"Dens{year}"}, inplace = True)
        
    # Now get the mean over all years:
    denscols = [col for col in df if col.startswith('Dens')]
    df["MeanDens"] = df[denscols].mean(axis = 1)
    totalcols = [col for col in df if col.startswith('Total')]
    df["MeanTotal"] = df[totalcols].mean(axis = 1)

    return df


# def ReadMODIS_MCD64A1(csv_path):
#     """
    

#     Parameters
#     ----------
#     csv_path : str
#         Path to .csv file that results from the Google Earth Engine Script:
#             https://code.earthengine.google.com/84d4275b0e18ab06912db61f430259ac

#     Returns
#     -------
#     pd.DataFrame
#         DataFrame with total burned area per year and the mean, stdev and ba_cv 
#         of the annual burned area per NUTS3 geometry in the European Fire Emissions Database

#     """
#     # Import the csv file as a pd.DataFrame
#     df = pd.read_csv(csv_path, delimiter = ',')
    
#     # Remove unneccesary columns
#     df.drop(df.columns[len(df.columns)-1], axis=1, inplace=True)
#     del df['system:index']
    
#     # Get the area from m2 to km2
#     df['sum'] = df['sum'] * 1e-06
    
#     # Create new df with all unique NUTS regions
#     ba_df = df['NUTS_ID'].drop_duplicates()
    
#     # Now filter all annual data per column
#     for year in range(int(df['year'].min()), int(df['year'].max())+1):
#         year_df = df.loc[(df["year"] == year)]
#         year_df = year_df[["NUTS_ID", "sum"]]
#         ba_df = pd.merge(ba_df, year_df, on=['NUTS_ID'])
#         ba_df = ba_df.rename(columns={'sum': str(year)})
    
#     # Calculate the coefficient of variation
#     ba_df['mean'] = ba_df.mean(axis = 1)
#     ba_df['stdev'] = ba_df.std(axis = 1)
#     ba_df['BA_CV'] = np.divide(ba_df['stdev'], ba_df['mean'])#, 
#                                #out = np.zeros(ba_df['stdev'].shape, dtype=float), where = ba_df['mean'] != 0)
    
#     return ba_df


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

