# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 16:09:40 2021

@author: Jasper Dijkstra

"""

# Built-in imports
#import os
import pandas as pd

# Import variables from Main Script
#from __main__ import wdir

# Local Imports
import VariableObjects as init
import ProcessData as pr


def GenerateIVS(zone_path, geo_name, prefix = "CNTR"):
    # 1. Generating Independent Variable Objects, from independent variable datasets
    iv1 = init.IndependentVariable(ID = 1, 
                                    name = 'Human Land Impact',
                                    author = 'Jacobsen et al. 2019',
                                    filename = f'{prefix}_LowImpactLand_Stats.xls', 
                                    source = 'https://doi.org/10.1038/s41598-019-50558-6',
                                    units = "% of Area subject to human impact",
                                    in_zones_shp = zone_path,
                                    value_field = geo_name)
    
    iv2 = init.IndependentVariable(ID = 2,
                                    name = "Altitude", 
                                    author = "USGS", 
                                    filename = f"{prefix}_DEM_Stats.xls", 
                                    source = "https://pubs.usgs.gov/of/2011/1073/pdf/of2011-1073.pdf", 
                                    units = "meter",
                                    in_zones_shp = zone_path,
                                    value_field = geo_name)
    
    iv3 = init.IndependentVariable(ID = 3, 
                                    name = "Population Density", 
                                    author = "Eurostat", 
                                    filename = "PopDensPerNUTS3.tsv", 
                                    source = "https://ec.europa.eu/eurostat/web/products-datasets/-/demo_r_d3dens",
                                    units = "Total Population / Area",
                                    in_zones_shp = zone_path,
                                    value_field = geo_name)
    
    iv5 = init.IndependentVariable(ID = 5, 
                                    name = "Tree Cover Density", 
                                    author = "EEA Copernicus Land Monitoring Service - 2018", 
                                    filename = f"{prefix}_TreeCoverDensity_Stats.xls", 
                                    source = "https://land.copernicus.eu/pan-european/high-resolution-layers/forests/tree-cover-density/status-maps/tree-cover-density-2018",
                                    units = "% of Area covered by trees",
                                    in_zones_shp = zone_path,
                                    value_field = geo_name)
    
    iv6 = init.IndependentVariable(ID = 6, 
                                    name = "BA Coefficient of Variation", 
                                    author = "NASA - MODIS MCD64A1", 
                                    filename = "BaCvPerStats.csv", 
                                    source = f"{prefix}_ImportingMODIS.py",
                                    units = "",
                                    in_zones_shp = zone_path,
                                    value_field = geo_name)
    
    iv7 = init.IndependentVariable(ID = 7, 
                                    name = "Terrain Ruggedness Index", 
                                    author = "Riley et al. 1999", 
                                    filename = f"{prefix}_DEM_Stats.xls", 
                                    source = r"https://download.osgeo.org/qgis/doc/reference-docs/Terrain_Ruggedness_Index.pdf",
                                    units = "",
                                    attribute = "STD",
                                    in_zones_shp = zone_path,
                                    value_field = geo_name)
    
    iv_list = [iv1, iv2, iv3, iv5, iv6, iv7] # list with independent variables

    return iv_list



def PredictOtherZones(zone_path, value_field, estimator_n, estimator_ba, prefix = "CNTR",
                      x_n = ["Altitude", "Population Density", "Terrain Ruggedness Index", "Tree Cover Density"],
                      x_ba = ["Altitude", "BA Coefficient of Variation", "Human Land Impact", "Population Density", "Terrain Ruggedness Index"]):
    """
    Predict 

    Parameters
    ----------
    zone_path : str
        Path to shapefile containing zones.
    value_field : str
        Name of the column containing unique geographic zone ID (e.g. NUTS_ID).
    estimator_n : sklearn.ensemble._forest.RandomForestRegressor
        RandomForest estimator to estimate the ratio of number of fires.
    estimator_ba : sklearn.ensemble._forest.RandomForestRegressor
        RandomForest estimator to estimate the ratio of burned area.
    prefix : str, optional
        Prefix of filenames of independent variables to look for (mostly .xls files). The default is "CNTR".
    x_n : list, optional
        Names of independent variables used by the RandomForest estimator to estimate Fire incidence. 
        The default is ["Altitude", "Population Density", "Terrain Ruggedness Index", "Tree Cover Density"].
    x_ba : list, optional
        Names of independent variables used by the RandomForest estimator to estimate BA. 
        The default is ["Altitude", "BA Coefficient of Variation", "Human Land Impact", "Population Density", "Terrain Ruggedness Index"].

    Returns
    -------
    df : pandas.DataFrame
        DatFrame containing the predicted fire incidence and burned area for all zones in the zones shapefile

    """
    # Read New Zones Shapefile
    from arcgis.features import GeoAccessor, GeoSeriesAccessor # enables to create spatial dataframe
    shp = pd.DataFrame.spatial.from_featureclass(zone_path)
    shp = pd.DataFrame(shp.drop(columns='SHAPE')) # Keep DataFrame without Geometries
    
    # Initialise all independent variable's
    iv_list = GenerateIVS(zone_path, value_field, prefix = prefix)
    
    # Summarize data into a DataFrame
    df, headers = pr.AnalysisDataFrame(shp, iv_list, geo_name = value_field)
    #valid_vars = [ i for i in headers if i not in ["NUTS_ID", "N_RATIO_Lightning", "BA_RATIO_Lightning"]] # Specify vars containing useable (numerical) data
    
    # Now predict using the Random Forest Estimators
    df["n_hum_hat"] = estimator_n.predict(df[x_n])
    df["ba_hum_hat"] = estimator_ba.predict(df[x_ba])
    
    df = pr.QuantifyBA(Predictions_df = df, 
                       MODIS_df = iv_list[4].metadata[[value_field, "mean"]], 
                       pred_att = "ba_hum_hat",
                       MODIS_att = "mean", 
                       geo_name = value_field)
    
    return df



