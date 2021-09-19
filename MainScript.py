# -*- coding: utf-8 -*-
"""
Created on Mon May 10 13:47:06 2021

@author: Jasper Dijkstra

This model imports data of several independent variables and then uses a multivariate regression analysis to compare it to the dependend variable: 
    European Fire Data from the European Fire Database (https://effis.jrc.ec.europa.eu/)


"""

# Built-in imports
import os

# Global Variables, that should work in all local scripts
wdir = os.path.join(r'C:\Users\jaspd\Desktop\AM_1265_Research_Project\02ArcGIS\01_ArcGIS_Project') # Working Directory

# Local Imports
import ClassObjects as init
import ProcessData as pr
import RandomForest as forest
import PlottingData as plot

# Behaviour Settings:
fig_wdir = r"C:\Users\jaspd\Google Drive\VU\AM_1265_Research_Project_Earth_And_Climate\02_Report\Figures"

determine_correlations = True
plot_correlations = True

perform_random_forest = True


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

# # Generated test dataset
# iv999 = init.IndependentVariable(ID = 999, 
#                                 name = 'Fictional', 
#                                 author = 'Jasper Dijkstra', 
#                                 filename = 'RandomTestData.xls', 
#                                 source = 'xxxxxx')


# 3. Sort input data to prepare for analysis:
iv_list = [iv1, iv2, iv3, iv4, iv5, iv6, iv7]#, iv999] # list with independent variables
df, headers = pr.AnalysisDataFrame(fires, iv_list) # Initiate DataFrame to be used for the analysis
valid_vars = [ i for i in headers if i not in ["NUTS_ID", "N_RATIO_Lightning", "BA_RATIO_Lightning"]] # Specify vars containing useable (numerical) data



#%%
# ========================================
# Analysis
# ========================================
# 1. Determine correlations and statistical significance (alpha < 0.05)
spearman_p, pearson_p, spearman_corr, pearson_corr = pr.CorrMatrix(df[valid_vars]) # Generate correlation matrices for all variables
spearman_corr[spearman_p > 0.05] = float('nan') # Remove non-significant correlations
pearson_corr[pearson_p > 0.05] = float('nan') # Remove non-significant correlations


# 2. Perform a Random Forest Analysis
if perform_random_forest:
    # If required, the data can be normalized
    # df2 = normalize(df, normlist)
    # for i in normlist:
    #     df = df.assign(**{i:df2[i]})
    y = df['N_RATIO_Human'] # df column with dependent variable
    #x = df.iloc[:,5:] # df columns with predictor variables
    #x = df[["Human Land Impact", "Altitude", "Population Density", "Tree Cover Density", "BA Coefficient of Variation", "Terrain Ruggedness Index"]]
    #x = df[['Population Density', 'Lightning Flashes per km2', 'Tree Cover Density', 'Terrain Ruggedness Index', 'Altitude']]
    
    # "Altitude"
    # "BA Coefficient of Variation"
    # "Human Land Impact"
    # "Population Density"
    # "Terrain Ruggedness Index"
    # "Tree Cover Density"
    
    # The best options:
    #x = df[["Altitude", "Population Density", "Terrain Ruggedness Index"]]
    #x = df[["Altitude", "BA Coefficient of Variation", "Population Density", "Terrain Ruggedness Index"]]
    x = df[["Altitude", "Population Density", "Terrain Ruggedness Index", "Tree Cover Density"]]
    #x = df[["Altitude", "BA Coefficient of Variation", "Population Density", "Terrain Ruggedness Index", "Tree Cover Density"]]


    
    # Initialising a Random Forest Analysis
    rfm = forest.RandomForest(x = x, y = y, labels = x.columns.to_list(), 
                       n_trees = 1000, # Number of trees to consider in default forest
                       test_size= 0.3, # Size (%) of the test data
                       #random_state=42, # Define random_state for reproducibility
                       scoring='explained_variance' # Scoring method to optimize
                       )
    
    # Tune parameters with randomized grid search
    # n_param_samples = amount of random samples to draw
    rfm.RandomizedGridSearch(n_param_samples = 50)
    
    # Narrow down parameters even further, using Grid Search
    rfm.GridSearch()
    #rfm.GridSearch(init_params='self') # redo a grid search, using its own optimal parameters
    
    #final_importance = rfm.GridSearch_Importances


#%% Create Plots

# 1: Plot Correlation Matrices
labels = ["Fire Incidence\n Ratio", "Burned Area\n Ratio", "Human Land\n Impact", "Mean Altitude", 
          "Population Density", "Lightning Flashes\n per km2", "Tree Cover\n Density", "Burned Area Coeff.\n of Variation", "Terrain Ruggedness\n Index"]

plot.CorrelationMatrix(data = pearson_corr, labels = labels, 
                       save_path = os.path.join(fig_wdir + os.path.sep + "Figx_CorrMatrrix_Pearson.png"))
plot.CorrelationMatrix(data = spearman_corr, labels = labels, 
                       save_path = os.path.join(fig_wdir + os.path.sep + "Figx_CorrMatrrix_Spearman.png"))


# Plot 6 x 2 Correlation Scatter Plots
xitems = ["N_RATIO_Human", "BA_RATIO_Human"]
yitems = ["Human Land Impact", "Altitude", "Population Density", "Tree Cover Density", "BA Coefficient of Variation", "Terrain Ruggedness Index"]

xlabels = ["Fire Incidence Ratio", "Burned Area Ratio"]
ylabels = ["Human Land \n Impact", "Mean Altitude", "Population Density", "TCD", "BA Coef. of \n Variation", "TRI"]

plot.CorrelationPlots(data = df, corr_idx = spearman_corr, xitems = xitems, yitems = yitems, 
                 save_path = os.path.join(fig_wdir + os.path.sep + "Figx_CorrelationPlots.png"), 
                 xlabels = xlabels, ylabels = ylabels)


