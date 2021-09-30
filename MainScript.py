# -*- coding: utf-8 -*-
"""
Created on Mon May 10 13:47:06 2021

@author: Jasper Dijkstra

This model imports data of several independent variables and then uses a Random Forest analysis to compare it to the dependend variable: 
    European Fire Data from the European Fire Database (https://effis.jrc.ec.europa.eu/)

Please read the projects' wiki on GitHub (https://github.com/jasper-dijkstra/MSc_ResearchProject/wiki) on how to build the
directory structure required for the model to import all data

In this script, only change the wdir variable (line 25) and those specified under "Define Input Data / Variables"

"""
# ===============================
# Import Libraries
# ===============================
# Built-in imports
import os
import pandas as pd
from datetime import datetime

# Define Working Firectory, that should work in all local scripts
wdir = os.path.join(r'C:\Users\jaspd\Desktop\AM_1265_Research_Project\02ArcGIS\01_ArcGIS_Project') # Working Directory

# Local Imports
import VariableObjects as init
import ProcessData as pr
import RandomForest as forest
import PredictOtherZones as predict
import PlottingData as plot

# ===============================
# Specify Functions
# ===============================
def SortData(fires, iv_list):
    # Sort input data to prepare for analysis:
    df, headers = pr.AnalysisDataFrame(fires.data[['NUTS_ID', 'N_RATIO_Human', 'N_RATIO_Lightning', 'BA_RATIO_Human', 'BA_RATIO_Lightning']],
                                       iv_list) # Initiate DataFrame to be used for the analysis
    valid_vars = [ i for i in headers if i not in ["NUTS_ID", "N_RATIO_Lightning", "BA_RATIO_Lightning"]] # Specify vars containing useable (numerical) data
    
    return df, valid_vars


def DetermineCorrelations(df, valid_vars):
    # Determine correlations and statistical significance (alpha < 0.05)
    spearman_p, pearson_p, spearman_corr, pearson_corr = pr.CorrMatrix(df[valid_vars]) # Generate correlation matrices for all variables
    spearman_corr[spearman_p > 0.05] = float('nan') # Remove non-significant correlations
    pearson_corr[pearson_p > 0.05] = float('nan') # Remove non-significant correlations
    
    return spearman_p, pearson_p, spearman_corr, pearson_corr


def RandomForestAnalysis(x, y, param_dict = None):
    """Instantiate Random Forest Analysis"""
    rfm = forest.RandomForest(x = x, 
                              y = y, 
                              labels = x.columns.to_list(), 
                              param_dict =  param_dict, # parameter values that have proven to be effective
                              test_size= 0.3, # Size (%) of the test data
                              #random_state=42, # Define random_state for reproducibility
                              scoring='explained_variance') # Scoring method to optimize
    
    if not param_dict: # If no parameter dict has been provided, determine optimal parameters automatically
        rfm.RandomizedGridSearch(n_param_samples = 50)  # Tune parameters with randomized grid search, n_param_samples = amount of random samples to draw
        rfm.GridSearch() # Narrow down parameters even further, using Grid Search
        #rfm.GridSearch(init_params='self') # redo a grid search, using its own optimal parameters
    
    # Determine estimator to use:
    if hasattr(rfm, 'GridSearch_Estimator'):
        estimator = rfm.GridSearch_Estimator
    elif hasattr(rfm, 'RandomGridSearch_Estimator'):
        estimator = rfm.RandomGridSearch_Estimator
    else:
        estimator = rfm.DefaultForest
    
    return rfm, estimator


# ===============================
# Define Input Data / Variables
# ===============================
t0 = datetime.now() # Register starting time of the model 

# Directory where output figures should be stored
fig_wdir = r"G:\Mijn Drive\VU\AM_1265_Research_Project_Earth_And_Climate\02_Report\Figures"

predict_other_zones = True # If true, fire ratios for other zones than NUTS3 will be estimated as well.
create_plots = False  # If true, data plots will be generated.

# Reading the fire data shp as dependent variable object (specify filepath):
fires = init.DependentVariable(filepath = os.path.join(wdir + os.path.sep + r"a0Data\b02Shapes\NUTS_fire2.shp"))

# 2. Generating Independent Variable Objects, from independent variable datasets (specify function arguments)
iv1 = init.IndependentVariable(ID = 1, 
                                name = 'Human Land Impact',
                                author = 'Jacobsen et al. 2019',
                                filename = 'LowImpactLand_NUTS3_Stats.xls', 
                                source = 'https://doi.org/10.1038/s41598-019-50558-6',
                                units = "% of Area subject to human impact")

iv2 = init.IndependentVariable(ID = 2,
                                name = "Altitude", 
                                author = "USGS", 
                                filename = "DEM_NUTS3_Stats.xls", 
                                source = "https://pubs.usgs.gov/of/2011/1073/pdf/of2011-1073.pdf", 
                                units = "meter")

iv3 = init.IndependentVariable(ID = 3, 
                                name = "Population Density", 
                                author = "Eurostat", 
                                filename = "PopDensPerNUTS3.tsv", 
                                source = "https://ec.europa.eu/eurostat/web/products-datasets/-/demo_r_d3dens",
                                units = "Total Population / Area")

iv4 = init.IndependentVariable(ID = 4, 
                                name = "Lightning Flashes per km2", 
                                author = "GHRC - LIS/OTD 0.5 Degree HRFC", 
                                filename = "LightnigStrikes_NUTS3_Stats.xls", 
                                source = "http://dx.doi.org/10.5067/LIS/LIS-OTD/DATA302",
                                units = "Mean amount of lightning Flashes per km2 per NUTS area")

iv5 = init.IndependentVariable(ID = 5, 
                                name = "Tree Cover Density", 
                                author = "EEA Copernicus Land Monitoring Service - 2018", 
                                filename = "TreeCoverDensity_NUTS3_Stats.xls", 
                                source = "https://land.copernicus.eu/pan-european/high-resolution-layers/forests/tree-cover-density/status-maps/tree-cover-density-2018",
                                units = "% of Area covered by trees")

iv6 = init.IndependentVariable(ID = 6, 
                                name = "BA Coefficient of Variation", 
                                author = "NASA - MODIS MCD64A1", 
                                filename = "BaCvPerNUTS.csv", 
                                source = "ImportingMODIS.py",
                                units = "")

iv7 = init.IndependentVariable(ID = 7, 
                                name = "Terrain Ruggedness Index", 
                                author = "Riley et al. 1999", 
                                filename = "DEM_NUTS3_Stats.xls", 
                                source = r"https://download.osgeo.org/qgis/doc/reference-docs/Terrain_Ruggedness_Index.pdf",
                                units = "",
                                attribute = "STD")

# ===============================
# Analysis
# ===============================
iv_list = [iv1, iv2, iv3, iv4, iv5, iv6, iv7] # list all defined independent variables

df, valid_vars = SortData(fires, iv_list) # Sort required data to pd.DataFrame
spearman_p, pearson_p, spearman_corr, pearson_corr = DetermineCorrelations(df, valid_vars) # Determine data correlations

# ========================================
# Perform Random Forest Analysis & Make predictions for Fire Incidence

n_forest_params =  {"bootstrap" : True,
                    "max_depth" : 80,
                    "max_features" : "log2",
                    "min_samples_leaf" : 5,
                    "min_samples_split" : 12,
                    "n_estimators" : 1500}

y = df['N_RATIO_Human'] # df column with dependent variable
x = df[["Altitude", "Population Density", "Terrain Ruggedness Index", "Tree Cover Density"]]

rfm_n, estimator_n = RandomForestAnalysis(x, y, param_dict = n_forest_params)
n_hat_nuts3 = pr.PredictNUTSZone(fires, iv_list, y.name, x, estimator = estimator_n)

# ========================================
# Perform Random Forest Analysis & Make predictions for Burned Area & Quantify (km2) this with MODIS data

ba_forest_params =  {"bootstrap" : True,
                     "max_depth" : 20,
                     "max_features" : "log2",
                     "min_samples_leaf" : 5,
                     "min_samples_split" : 12,
                     "n_estimators" : 500}

y = df['BA_RATIO_Human'] # df column with dependent variable
x = df[["Altitude", "BA Coefficient of Variation", "Human Land Impact", "Population Density", "Terrain Ruggedness Index"]]

rfm_ba, estimator_ba = RandomForestAnalysis(x, y, param_dict = ba_forest_params)
ba_hat_nuts3 = pr.PredictNUTSZone(fires, iv_list, y.name, x, estimator = estimator_ba)

# Quantify BA With MODIS
ratioBA = fires.data_with_nan[["NUTS_ID", "BA_RATIO_Human", "NUTS_Area"]].copy() # Get BA ratios in new df
ratioBA = pd.merge(ratioBA, ba_hat_nuts3, on = ["NUTS_ID"], how = 'outer') # Append the predicted ones to it
ratioBA["BA_RATIO_Human"] = ratioBA["BA_RATIO_Human"].fillna(ratioBA["Exp_BA_RATIO_Human"]) # Fill observation gaps with predictions
ratioBA = ratioBA.drop(["Exp_BA_RATIO_Human"], axis=1) # Drop predictions column
ratioBA = pr.QuantifyBA(Predictions_df = ratioBA, MODIS_df = iv6.metadata[["NUTS_ID", "mean"]], 
                        MODIS_att = "mean", geo_name = "NUTS_ID")

# ========================================
# Export Predicted Fire Incidence & Burned Area to csv
out_csv = fires.data_with_nan[["NUTS_ID", "CNTR_CODE", "NUTS_Area", "N_RATIO_Human", "BA_RATIO_Human"]]
out_csv = pd.merge(out_csv, n_hat_nuts3, on=['NUTS_ID'], how = 'outer') # Append n_ratios to all data
out_csv = pd.merge(out_csv, ratioBA, on=['NUTS_ID'], how = 'outer') # Also append ba_ratios to it

out_csv.to_csv(os.path.join(wdir + os.path.sep + r"a0Data\b03ExcelCSV\FireRatios_Predicted_NUTS3.csv"),
               sep = ';',
               decimal = '.')
print("Saved Predicted Fire Incidence & Burned Area to: \n {} \n".format(os.path.join(wdir + os.path.sep + r'a0Data\b03ExcelCSV\FireRatios_Predicted_NUTS3.csv')))



if predict_other_zones:
    country_df = predict.PredictOtherZones(os.path.join(wdir + os.path.sep + r"a0Data\b02Shapes\country_shps.shp"), 
                                           value_field = "CNTR_CODE", 
                                           estimator_n = estimator_n, 
                                           estimator_ba = estimator_ba)

    country_df.to_csv(os.path.join(wdir + os.path.sep + r"a0Data\b03ExcelCSV\FireRatios_Predicted_Country.csv"),
                   sep = ';',
                   decimal = '.')

    print("Saved Predicted Country level Fire Incidence & Burned Area to: \n {} \n".format(wdir + os.path.sep + r"a0Data\b03ExcelCSV\FireRatios_Predicted_Country.csv"))



# ========================================
# Create Plots (if desired)
# ========================================
if create_plots:
    # Plot Correlation Matrices
    labels = ["Fire Incidence\n Ratio", "Burned Area\n Ratio", "Human Land\n Impact", "Mean Altitude", 
              "Population Density", "Lightning Flashes\n per km2", "Tree Cover\n Density", "Burned Area Coeff.\n of Variation", "Terrain Ruggedness\n Index"]
    
    plot.CorrelationMatrix(data = pearson_corr, labels = labels, 
                           save_path = os.path.join(fig_wdir + os.path.sep + "Figx_CorrMatrix_Pearson.png"))
    plot.CorrelationMatrix(data = spearman_corr, labels = labels, 
                           save_path = os.path.join(fig_wdir + os.path.sep + "Figx_CorrMatrix_Spearman.png"))
    
    
    # Plot 6 x 2 Correlation Scatter Plots
    xitems = ["N_RATIO_Human", "BA_RATIO_Human"]
    yitems = ["Human Land Impact", "Altitude", "Population Density", "Tree Cover Density", "BA Coefficient of Variation", "Terrain Ruggedness Index"]
    
    xlabels = ["Fire Incidence Ratio", "Burned Area Ratio"]
    ylabels = ["Human Land \n Impact", "Mean Altitude", "Population Density", "TCD", "BA Coef. of \n Variation", "TRI"]
    
    plot.CorrelationPlots(data = df, corr_idx = spearman_corr, xitems = xitems, yitems = yitems, 
                         save_path = os.path.join(fig_wdir + os.path.sep + "Figx_CorrelationPlots.png"), 
                         xlabels = xlabels, ylabels = ylabels)
    
    
    # Plot Bar Charts with relative importance of variables
    plot.FeatureImportance(ForestRegressor = estimator_n, 
                           labels = rfm_n.labels, 
                           save_path = os.path.join(fig_wdir + os.path.sep + 'Figx_RelativeImportanceBars'),
                           ForestRegressor2 = estimator_ba, 
                           labels2 = rfm_ba.labels)
    
    
    # Plot the observations and predictions of the test sets of the random forest model 
    plot.RandomForestPerformance(rfm_1 = rfm_n, rfm_1_estimator = estimator_n,
                                 rfm_2 = rfm_ba, rfm_2_estimator = estimator_ba, 
                                 save_path = os.path.join(fig_wdir + os.path.sep + r'Figx_PerformanceScatter.png'))

print(f"Total time elapsed: {datetime.now()-t0}")
