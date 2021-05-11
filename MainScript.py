# -*- coding: utf-8 -*-
"""
Created on Mon May 10 13:47:06 2021

@author: Jasper Dijkstra

This script runs 

"""

# Imports
import os


# Setting up the environment
wdir = os.path.join(r'C:\Users\jaspd\Desktop\AM_1265_Research_Project\02ArcGIS\01_ArcGIS_Project') # Working DIrectory
data_dir = os.path.join(wdir + os.path.sep +  r'a0Data\b03ExcelCSV') # Relative path to input csv files

# Local Imports
import ClassObjects as init
#import Read_Data as read
#import Utilities as utils




#init.GlobalVars(wdir, data_dir) # allowing the use of the abovementioned variables in other scripts as well

# Import the fire data 
fires = init.DependentVariable(filepath = os.path.join(wdir + os.path.sep + r"a0Data\b02Shapes\NUTS_fire2.shp"))


#%% Generating Independent Variable Objects, from independent variables

iv1 = init.IndependentVariable(ID = 1, 
                               name = 'Low Impact Land',
                               author = 'Jacobsen et al. 2019',
                               filename = 'LowImpactLand_NUTS3_Stats.xls', 
                               source = 'https://doi.org/10.1038/s41598-019-50558-6')

# Generate test dataset
iv2 = init.IndependentVariable(ID = 999, 
                               name = 'Fictional', 
                               author = 'Jasper Dijkstra', 
                               filename = 'RandomTestData.xls', 
                               source = 'xxxxxx')

#%% Use the data of the independent variable objects in a multivariate analysis

#%% ===== Defining Variables =====
#FireData_shp = os.path.join(data_dir + os.path.sep + r'\b02Shapes\NUTS_fire2.shp')


dep = fires.data
indep1 = iv1.data
indep2 = iv2.data



#%% ===== Importing Datasets =====

# # Read Main Shapefile containing the fire data per NUTS3 region
#FireData = read.SHP_to_gpdFD(FireData_shp)
