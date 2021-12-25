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
import numpy as np
from datetime import datetime

# Define Working Firectory, that should work in all local scripts
wdir = os.path.join(r'C:\Users\jaspd\Desktop\AM_1265_Research_Project\02ArcGIS\01_ArcGIS_Project') # Working Directory

# Local Imports
import VariableObjects as init
import ProcessData as process
import RandomForest as forest
import PlottingData as plot


# ===============================
# Specify Functions
# ===============================
def RandomForestAnalysis(x, y, param_dict = None, predict_arr=None):
    """Instantiate Random Forest Analysis"""
    rfm = forest.RandomForest(x = x, 
                              y = y, 
                              labels = x.columns.to_list(), 
                              param_dict =  param_dict, # parameter values that have proven to be effective
                              test_size= 0.3, # Size (%) of the test data
                              #random_state=42, # Define random_state for reproducibility
                              scoring='explained_variance', # Scoring method to optimize
                              predict_arr = predict_arr)
    
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


def CheckAllVariableCombinations(analysis_obj, file_out):
    """
    
    This function iterates over all possible parameter combinations, performs a random forest analysis,
    and saves its results (r-squared, mean absolute error, explained variance and mean standard deviation) in a csv file.
    
    input: analysis_obj (process.AnalysisObject), file_out (str)
    """
    all_items = ['Human Land Impact', "Altitude", "Population Density", "Tree Cover Density", "BA Coefficient of Variation", "Terrain Ruggedness Index"]
    
    import itertools
    
    all_combis = []
    
    for i in range(2, len(all_items)+1):
        for subset in itertools.combinations(all_items, i):
            all_combis.append(list(subset))
            
    def RFM_LOOP(all_combis, analysis_obj, fileout):
        
        performace_df = pd.DataFrame(index = range(0, len(all_combis)), columns = ['variables', 'R2', 'MAE', 'exp_var', 'trees_std'])
        
        for i, xlabels in enumerate(all_combis):
            print(xlabels)
            rfm, estimator = RandomForestAnalysis(x = analysis_obj.independent[xlabels], 
                                                  y = analysis_obj.dependent.drop("NUTS_ID", axis=1), 
                                                  param_dict = None)
            
            performace_df['variables'][i] = xlabels
            performace_df['R2'][i] = rfm.RandomGridSearch_Performance["R-squared"]
            performace_df['MAE'][i] = rfm.RandomGridSearch_Performance["Mean Absolute Error"]
            performace_df['exp_var'][i] = rfm.RandomGridSearch_Performance["Explained Variance"]
            
            mean, std = fi_rfm.UncertaintyEstimate(estimator, analysis_obj.independent[xlabels])
            performace_df['trees_std'][i] = np.mean(std)
            
            # to csv, delimiter ;
            performace_df.to_csv(fileout, sep = ';')
    
    RFM_LOOP(all_combis, analysis_obj, file_out)

    return


def CompareScores(x, y, params):
    
    # Create list with all xlabels used, and all xlabels without one
    xlabels = list(x.columns)
    combis = [xlabels]
    for i in range(len(xlabels)):
        c = [p for p in xlabels if p != xlabels[i]]
        combis.append(c)
    
    # Now, for each one, determine the best R2, and note this down
    out_dict = {}
    for n, i in enumerate(combis):
        rfm, estimator = RandomForestAnalysis(x = x[i], y = y, param_dict = params) # Random Forest Ananlysis
        unc = np.std(rfm.score_uncertainty) # Standard Deviation of different R2 scores
        score = rfm.DefaultForest_performance["R-squared"]
        out_dict[n] = {"xlabels" : i, 'r2' : score, 'uncertainty' : unc}
        
    return out_dict

# ===============================
# Define Input Data / Variables
# ===============================
t0 = datetime.now() # Register starting time of the model 

# Directory where output figures should be stored
fig_wdir = r"G:\Mijn Drive\VU\AM_1265_Research_Project_Earth_And_Climate\02_Report\Figures"

create_plots = True  # If true, data plots will be generated.

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
# Analysis: Fire Incidence
# ===============================

fire_incidence = process.AnalysisObject(dependent_variable = fires.data[["NUTS_ID", "Ln_N"]], 
                                        independent_variables = [iv1, iv2, iv3, iv4, iv5, iv6, iv7],
                                        raw_data = fires.data_with_nan)


fi_xlabels = ["Altitude", "BA Coefficient of Variation", "Human Land Impact", "Population Density"]
fi_forest_params =  {"bootstrap" : True,
                    "max_depth" : 10,
                    "max_features" : "log2",
                    "min_samples_leaf" : 1,
                    "min_samples_split" : 5,
                    "n_estimators" : 600}

# Make sure all indices are sorted! 
fi_x = fire_incidence.independent[["NUTS_ID"] + fi_xlabels].set_index("NUTS_ID")
fi_y = fire_incidence.dependent.set_index("NUTS_ID")
fi_xy = pd.merge(fi_x, fi_y, left_index=True, right_index=True)

fi_rfm, fi_estimator = RandomForestAnalysis(x = fi_xy[fi_xlabels], 
                                            y = fi_xy.iloc[:,-1], 
                                            param_dict = fi_forest_params,
                                            predict_arr = fire_incidence.unknown_ratios[["NUTS_ID"] + fi_xlabels].set_index("NUTS_ID"))

# Get an estimate of the uncertainty
fi_unc = pd.DataFrame(fire_incidence.unknown_ratios["NUTS_ID"])
fi_unc["std"] = fi_rfm.std_predictions

fire_incidence.Predict(fi_estimator, fi_xlabels) # Since we have a RFM estimator, we can predict gaps in our data

# Export results to a csv file
fire_incidence.ExportToCSV(os.path.join(wdir + os.path.sep + r"a0Data\b03ExcelCSV\Predicted_FireIncidence.csv"), extra_cols = fi_unc)

# ===============================
# Analysis: Burned Area
# ===============================

burned_area = process.AnalysisObject(dependent_variable = fires.data[["NUTS_ID", "Ln_BA"]], 
                                     independent_variables = [iv1, iv2, iv3, iv4, iv5, iv6, iv7],
                                     raw_data = fires.data_with_nan)


ba_xlabels = ["Altitude", "BA Coefficient of Variation", "Human Land Impact", "Population Density"]
ba_forest_params =  {"bootstrap" : True,
                     "max_depth" : 40,
                     "max_features" : "log2",
                     "min_samples_leaf" : 1,
                     "min_samples_split" : 12,
                     "n_estimators" : 1000}

# Make sure all indices are sorted! 
ba_x = burned_area.independent[["NUTS_ID"] + ba_xlabels].set_index("NUTS_ID")
ba_y = burned_area.dependent.set_index("NUTS_ID")
ba_xy = pd.merge(ba_x, ba_y, left_index=True, right_index=True)

ba_rfm, ba_estimator = RandomForestAnalysis(x = ba_xy[fi_xlabels], 
                                            y = ba_xy.iloc[:,-1], 
                                            param_dict = ba_forest_params,
                                            predict_arr = burned_area.unknown_ratios[["NUTS_ID"] + ba_xlabels].set_index("NUTS_ID"))

# Get an estimate of the uncertainty
ba_unc = pd.DataFrame(burned_area.unknown_ratios["NUTS_ID"])
ba_unc["std"] = ba_rfm.std_predictions

burned_area.Predict(ba_estimator, ba_xlabels, modis_ba = iv6.metadata[["NUTS_ID", "mean"]]) # Since we have a RFM estimator, we can predict gaps in our data

# Export results to a csv file
burned_area.ExportToCSV(os.path.join(wdir + os.path.sep + r"a0Data\b03ExcelCSV\Predicted_BurnedArea.csv"), extra_cols = ba_unc)
burned_area.ExportToCSV(os.path.join(wdir + os.path.sep + r"a0Data\b03ExcelCSV\Predicted_BurnedArea_Country.csv"), geo_name = "CNTR_CODE")


# ========================================
# Create Plots (if desired)
# ========================================
if create_plots:
    # Initialise lists with names of all x an y items
    xitems = ["N_RATIO_Human", "BA_RATIO_Human"]
    yitems = ["Human Land Impact", "Altitude", "Population Density", "Lightning Flashes per km2", "Tree Cover Density", "BA Coefficient of Variation", "Terrain Ruggedness Index"]
    
    # Initialise df with all data required for most plots summarized
    df = pd.merge(fire_incidence.raw[["NUTS_ID", xitems[0]]].dropna(), 
                  burned_area.raw[["NUTS_ID", xitems[1]]].dropna(), on="NUTS_ID", how='inner')
    df = pd.merge(df, fire_incidence.independent, on = "NUTS_ID")
    
    # Create a correlation matrix of all data
    spearman_p, pearson_p, spearman_corr, pearson_corr = fire_incidence.DetermineCorrelations(df, xitems, yitems, confidence=0.95)#[2]
    
    
    # ====== Plot Correlation Matrices ======
    labels = ["Anthropogenic Fire \nIncidence Fraction", "Anthropogenic Burned \nArea Fraction", "Human Land\n Impact", "Mean Altitude", 
              "Population Density", "Lightning Flashes\n per km2", "Tree Cover\n Density", "Burned Area Coeff.\n of Variation", "Terrain Ruggedness\n Index"]
    
    plot.CorrelationMatrix(data = pearson_corr, labels = labels, 
                            save_path = os.path.join(fig_wdir + os.path.sep + "Figx_CorrMatrix_Pearson.jpg"))
    
    
    plot.CorrelationMatrix(data = spearman_corr, labels = labels, 
                            save_path = os.path.join(fig_wdir + os.path.sep + "Fig3_CorrMatrix_Spearman.jpg"))
    
    
    # ====== Plot 6 x 2 Correlation Scatter Plots ======
    xlabels = ["Anthropogenic Fire \nIncidence Fraction", "Anthropogenic Burned \nArea Fraction"]
    ylabels = ["Human Land \n Impact", "Mean \n Altitude (m)", "Population \n per $km^2$", "Tree Cover \n Density", "BA Coefficient \n of Variation", "Terrain Ruggedness \n Index"]
    yitems2 = yitems.copy()
    yitems2.remove("Lightning Flashes per km2")
    
    plot.CorrelationPlots(data = df, corr_idx = spearman_corr, xitems = xitems, yitems = yitems2, 
                          save_path = os.path.join(fig_wdir + os.path.sep + "Fig4_CorrelationPlots.jpg"), 
                          xlabels = xlabels, ylabels = ylabels)
    
    
    # ===== Plot Actual Test, vs Predicted Test data =====
    plot.RandomForestPerformance(fi_rfm, fi_estimator, ba_rfm, ba_estimator, 
                                  save_path = os.path.join(fig_wdir + os.path.sep + "Fig5_PerformanceScatter.jpg"))
    
    
    # ===== Plot Bar Charts with relative importance of variables =====
    plot.FeatureImportance(ForestRegressor = fi_estimator, 
                           labels = fi_rfm.labels, 
                           save_path = os.path.join(fig_wdir + os.path.sep + 'Fig6_RelativeImportanceBars.jpg'),
                           ForestRegressor2 = ba_estimator, 
                           labels2 = ba_rfm.labels)
    
    # ===== Plot Bar Charts model score and effects of removing variables =====
    fi_out = CompareScores(x = fi_xy[fi_xlabels], y = fi_xy.iloc[:,-1], params = fi_forest_params) # Check effects of varying variables to the model score
    ba_out = CompareScores(x = ba_xy[ba_xlabels], y = ba_xy.iloc[:,-1], params = ba_forest_params) # Check effects of varying variables to the model score
    
    plot.CompareModelScore(save_path = os.path.join(fig_wdir + os.path.sep + 'Fig7_ModelScoreBars.jpg'), 
                            out = fi_out,
                            out2 = ba_out)
    
    

print(f"Total time elapsed: {datetime.now()-t0}")


