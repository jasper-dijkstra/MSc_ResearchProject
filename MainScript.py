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
import PlottingData as plot

# ========================================
# INPUT DATA
# ========================================
#%% Reading the fire data shp as dependent variable object

fires = init.DependentVariable(filepath = os.path.join(wdir + os.path.sep + r"a0Data\b02Shapes\NUTS_fire2.shp"))
dv_attributes = fires.data_header
cntrs = fires.countries


#%% Generating Independent Variable Objects, from independent variable datasets

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




#%% Use the data of the independent variable objects in a multivariate analysis

iv_list = [iv1, iv2, iv3, iv4, iv5, iv6, iv7]#, iv999] # list with independent variables

df, headers = pr.AnalysisDataFrame(fires, iv_list) # !!! Initiate DataFrame to be used for the analysis

spearman_corr, pearson_corr = pr.CorrMatrix(df, skip = 1) # Generate correlation matrices for all variables
#mean_pearson = (np.sum(np.abs(np.array(pearson_corr))) - pearson_corr.shape[0]) / (np.product(pearson_corr.shape) - pearson_corr.shape[0])
#mean_spearman = (np.sum(np.abs(np.array(spearman_corr))) - spearman_corr.shape[0]) / (np.product(spearman_corr.shape) - spearman_corr.shape[0])


#%%

plot_correlations = False
plot_colinearity = False
plot_relationships = False    

fig_wdir = r"C:\Users\jaspd\Google Drive\VU\AM_1265_Research_Project_Earth_And_Climate\04_Notes\Images"

# Plot Pearson & Spearman Correlations:
if plot_correlations:
    plot.CorrelationMatrix(data = pearson_corr, headers = headers, title="Pearson Correlations", 
                            save_path=os.path.join(fig_wdir + r"\CorrMatrrix_Pearson.png"))
    plot.CorrelationMatrix(data = spearman_corr, headers = headers, title="Spearman Correlations", 
                            save_path=os.path.join(fig_wdir + r"\CorrMatrrix_Spearman.png"))


# Now Create Plots Of Interesting Combinations:

# Top 7 Colinear Relationships
if plot_colinearity:
    plot.Scatter(df[iv2.name], df[iv7.name], save_path= os.path.join(fig_wdir + r'\Plot_Altitude_x_TRI.png'))
    plot.Scatter(df[iv1.name], df[iv5.name], save_path= os.path.join(fig_wdir + r'\Plot_LIA_x_TCD.png'))
    plot.Scatter(df[iv4.name], df[iv7.name], save_path= os.path.join(fig_wdir + r'\Plot_Lightning_x_TRI.png'))
    plot.Scatter(df[iv2.name], df[iv4.name], save_path= os.path.join(fig_wdir + r'\Plot_Altitude_x_Lightning.png'))
    plot.Scatter(df[iv1.name], df[iv7.name], save_path= os.path.join(fig_wdir + r'\Plot_LIA_x_TRI.png'))
    plot.Scatter(df[iv6.name], df[iv3.name], save_path= os.path.join(fig_wdir + r'\Plot_BACV_x_PopDens.png'))
    plot.Scatter(df[iv6.name], df[iv4.name], save_path= os.path.join(fig_wdir + r'\Plot_BACV_x_Lightning.png'))

# Plot relationships between dependent and independent variables
if plot_relationships:
    plot.Scatter(df["N_RATIO_Human"], df["BA_RATIO_Human"], save_path= os.path.join(fig_wdir + r'\Plot_Nhum_x_BAhum.png'))
    
    plot.Scatter(df[iv1.name], df["N_RATIO_Human"], save_path= os.path.join(fig_wdir + r'\Plot_LIA_x_Nhum.png'))
    plot.Scatter(df[iv3.name], df["N_RATIO_Human"], save_path= os.path.join(fig_wdir + r'\Plot_PopDens_x_Nhum.png'))
    plot.Scatter(df[iv7.name], df["N_RATIO_Human"], save_path= os.path.join(fig_wdir + r'\Plot_TRI_x_Nhum.png'))

    plot.Scatter(df[iv1.name], df["BA_RATIO_Human"], save_path= os.path.join(fig_wdir + r'\Plot_LIA_x_BAhum.png'))
    plot.Scatter(df[iv3.name], df["BA_RATIO_Human"], save_path= os.path.join(fig_wdir + r'\Plot_PopDens_x_BAhum.png'))
    plot.Scatter(df[iv5.name], df["BA_RATIO_Human"],  save_path= os.path.join(fig_wdir + r'\Plot_TCD_x_BAhum.png'))


# def normalize(df, features_to_normalize):
#     result = df.copy()
#     result = result[features_to_normalize]
#     for feature_name in features_to_normalize:
#         max_value = df[feature_name].max()
#         min_value = df[feature_name].min()
#         result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
#     return result
# 
# #df2 = normalize(df, [""])
# 

