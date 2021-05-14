# -*- coding: utf-8 -*-
"""
Created on Mon May 10 13:47:06 2021

@author: Jasper Dijkstra

This model imports data of several independent variables and then uses a multivariate-
regression analysis to compare it to the dependend variable: European Fire Data from the 
European Fire Database (https://effis.jrc.ec.europa.eu/)


"""

# Built-in imports
import os

# Global Variables, that should work in all local scripts
wdir = os.path.join(r'C:\Users\jaspd\Desktop\AM_1265_Research_Project\02ArcGIS\01_ArcGIS_Project') # Working Directory
data_dir = os.path.join(wdir + os.path.sep +  r'a0Data\b03ExcelCSV') # Relative path to input csv files

# Local Imports
import ClassObjects as init
import MultivariateAnalysis as mv


#%% Reading the fire data shp

fires = init.DependentVariable(filepath = os.path.join(wdir + os.path.sep + r"a0Data\b02Shapes\NUTS_fire2.shp"))


#%% Generating Independent Variable Objects, from independent variable datasets

iv1 = init.IndependentVariable(ID = 1, 
                                name = 'Low Impact Land',
                                author = 'Jacobsen et al. 2019',
                                filename = 'LowImpactLand_NUTS3_Stats.xls', 
                                source = 'https://doi.org/10.1038/s41598-019-50558-6')

iv2 = init.IndependentVariable(ID = 2,
                               name = "Altitude", 
                               author = "USGS", 
                               filename = "DEM_NUTS3_Stats.xls", 
                               source = "xxxxxx")

# Generated test dataset
iv999 = init.IndependentVariable(ID = 999, 
                               name = 'Fictional', 
                               author = 'Jasper Dijkstra', 
                               filename = 'RandomTestData.xls', 
                               source = 'xxxxxx')




#%% Use the data of the independent variable objects in a multivariate analysis

iv_list = [iv1, iv2] # list with independent variables
attribute = "MEAN" # attribute to look at in analysis

df = mv.AnalysisDataFrame(fires, iv_list, attribute) # Generate DataFrame to be used for the analysis

# Here regression analysis
ba_n = "N_RATIO" # Which independent variable will be used: BA_RATIO or N_RATIO
















