# -*- coding: utf-8 -*-
"""
Created on Mon May 10 13:47:06 2021

@author: Jasper Dijkstra

This model imports data of several independent variables and then uses a multivariate regression analysis to compare it to the dependend variable: 
    European Fire Data from the European Fire Database (https://effis.jrc.ec.europa.eu/)


"""

# Built-in imports
import os
import pandas as pd

# Global Variables, that should work in all local scripts
wdir = os.path.join(r'C:\Users\jaspd\Desktop\AM_1265_Research_Project\02ArcGIS\01_ArcGIS_Project') # Working Directory

# Local Imports
import VariableObjects as init
import ProcessData as pr
import RandomForest as forest
import PredictOtherZones as predict
import PlottingData as plot


from datetime import datetime
t0 = datetime.now()

# Behaviour Settings:
fig_wdir = r"G:\Mijn Drive\VU\AM_1265_Research_Project_Earth_And_Climate\02_Report\Figures"


use_pre_defined_parameters = True # If false, parameters will be determined automatically
predict_other_zones = True # If true, fire ratios for other zones than NUTS3 will be estimated as well.
export_predictions = True # If true, predicted ratio's will be exported to a csv file

create_plots = False

# ========================================
# INPUT DATA
# ========================================
# 1. Reading the fire data shp as dependent variable object

fires = init.DependentVariable(filepath = os.path.join(wdir + os.path.sep + r"a0Data\b02Shapes\NUTS_fire2.shp"))
dv_attributes = fires.data_header
cntrs = fires.countries


# 2. Generating Independent Variable Objects, from independent variable datasets
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

# 3. Sort input data to prepare for analysis:
iv_list = [iv1, iv2, iv3, iv4, iv5, iv6, iv7] # list with independent variables
df, headers = pr.AnalysisDataFrame(fires.data[['NUTS_ID', 'N_RATIO_Human', 'N_RATIO_Lightning', 'BA_RATIO_Human', 'BA_RATIO_Lightning']],
                                   iv_list) # Initiate DataFrame to be used for the analysis
valid_vars = [ i for i in headers if i not in ["NUTS_ID", "N_RATIO_Lightning", "BA_RATIO_Lightning"]] # Specify vars containing useable (numerical) data


#%%
# ========================================
# Analysis
# ========================================
# 1. Determine correlations and statistical significance (alpha < 0.05)
spearman_p, pearson_p, spearman_corr, pearson_corr = pr.CorrMatrix(df[valid_vars]) # Generate correlation matrices for all variables
spearman_corr[spearman_p > 0.05] = float('nan') # Remove non-significant correlations
pearson_corr[pearson_p > 0.05] = float('nan') # Remove non-significant correlations


# ========================================
# 2. Generate a Random Forest Model & predict ratio's
if use_pre_defined_parameters: # Use pre-defined parameters
    n_forest_params =  {"bootstrap" : True,
                        "max_depth" : 80,
                        "max_features" : "log2",
                        "min_samples_leaf" : 5,
                        "min_samples_split" : 12,
                        "n_estimators" : 1500}
    
    ba_forest_params =  {"bootstrap" : True,
                         "max_depth" : 20,
                         "max_features" : "log2",
                         "min_samples_leaf" : 5,
                         "min_samples_split" : 12,
                         "n_estimators" : 500}
else: # Determine parameters as part of the process 
    n_forest_params = None
    ba_forest_params = None
    

# Start with Fire incidence ratio:
y = df['N_RATIO_Human'] # df column with dependent variable
x = df[["Altitude", "Population Density", "Terrain Ruggedness Index", "Tree Cover Density"]]

rfm_n = forest.RandomForest(x = x, y = y, labels = x.columns.to_list(), 
                            param_dict =  n_forest_params, # parameter values that have proven to be effective
                            test_size= 0.3, # Size (%) of the test data
                            #random_state=42, # Define random_state for reproducibility
                            scoring='explained_variance') # Scoring method to optimize
estimator_n = rfm_n.DefaultForest

if not use_pre_defined_parameters:
    rfm_n.RandomizedGridSearch(n_param_samples = 50)  # Tune parameters with randomized grid search, n_param_samples = amount of random samples to draw
    rfm_n.GridSearch() # Narrow down parameters even further, using Grid Search
    #rfm.GridSearch(init_params='self') # redo a grid search, using its own optimal parameters
    estimator_n = rfm_n.GridSearch_Estimator

# Predict at NUTS3 level
df_n_hat_nuts3 = pr.PredictNUTSZone(fires, iv_list, y.name, x, 
                                  estimator = estimator_n) # RandomForestRegressor


# Do the same for the Burned Area Ratio
y = df['BA_RATIO_Human']
x = df[["Altitude", "BA Coefficient of Variation", "Human Land Impact", "Population Density", "Terrain Ruggedness Index"]]

rfm_ba = forest.RandomForest(x = x, y = y, labels = x.columns.to_list(), 
                            param_dict =  ba_forest_params, # parameter values that have proven to be effective
                            test_size= 0.3, # Size (%) of the test data
                            #random_state=42, # Define random_state for reproducibility
                            scoring='explained_variance') # Scoring method to optimize
estimator_ba = rfm_ba.DefaultForest

if not use_pre_defined_parameters:
    rfm_ba.RandomizedGridSearch(n_param_samples = 50)  # Tune parameters with randomized grid search, n_param_samples = amount of random samples to draw
    rfm_ba.GridSearch() # Narrow down parameters even further, using Grid Search
    #rfm.GridSearch(init_params='self') # redo a grid search, using its own optimal parameters
    estimator_ba = rfm_ba.GridSearch_Estimator #!!! Automatically determine which estimator is present

# Predict at NUTS3 level
df_ba_hat_nuts3 = pr.PredictNUTSZone(fires, iv_list, y.name, x, 
                                   estimator = estimator_ba) # RandomForestRegressor


if predict_other_zones:
    country_df = predict.PredictOtherZones(os.path.join(wdir + os.path.sep + r"a0Data\b02Shapes\country_shps.shp"), 
                                           value_field = "CNTR_CODE", 
                                           estimator_n = estimator_n, 
                                           estimator_ba = estimator_ba)

    country_df.to_csv(os.path.join(wdir + os.path.sep + r"a0Data\b03ExcelCSV\FireRatios_Predicted_Country.csv"),
                   sep = ';',
                   decimal = '.')


if export_predictions:
    out_csv = fires.data_with_nan[["NUTS_ID", "CNTR_CODE", "NUTS_Area", "N_RATIO_Human", "BA_RATIO_Human"]]
    out_csv = pd.merge(out_csv, df_n_hat_nuts3, on=['NUTS_ID'], how = 'outer') # Append n_ratios to all data
    out_csv = pd.merge(out_csv, df_n_hat_nuts3, on=['NUTS_ID'], how = 'outer') # Also append ba_ratios to it
    
    out_csv.to_csv(os.path.join(wdir + os.path.sep + r"a0Data\b03ExcelCSV\FireRatios_Predicted_NUTS3.csv"),
                   sep = ';',
                   decimal = '.')
    




#%%
# ========================================
# Create Plots
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
