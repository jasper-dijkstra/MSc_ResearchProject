# -*- coding: utf-8 -*-
"""
Created on Wed May 26 15:44:01 2021

@author: jaspd

Script with functions to process input input data:
    - AnalysisDataFrame: Generates DataFrame that facilitates later analysis
    - CorrMatrix: Generates a pearson and spearman correlation matrix from input AnalysisDataFrame
    
"""
import re
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr

class AnalysisObject:
    
    def __init__(self, dependent_variable, independent_variables, raw_data):
        self.dependent = dependent_variable
        self.dependent_label = dependent_variable.iloc[:,-1].name
        self.iv_list = independent_variables
        self.independent, self.independent_labels = self.__IndependentVariablesDF__(geo_col = self.dependent.iloc[:,0])
        
        # Esatblish correlations, by creating correlation matrices
        self.spearman_p, self.pearson_p, self.spearman_corr, self.pearson_corr = self.DetermineCorrelations(
                                            df = self.dependent.merge(self.independent, on = self.dependent.iloc[:,0].name),
                                            ylabels = self.dependent_label, 
                                            xlabels = self.independent_labels)
        
        self.raw = raw_data # raw EFFIS shapefile data
        self.unknown_ratios = self.__UnknownRatios__() # df with nuts zones lacking effis data
        #self.unknown_country_ratios = self.__DetermineCountryRatios__() # df on country scale

        return
    
    
    def Predict(self, estimator, xlabels, modis_ba = None):
        """
        Predict ratio's of fire sources. When modis_ba is provided, also the burned area of each source, 
        and the ratios on a national level are calculated.

        Parameters
        ----------
        estimator : ensemble._forest.RandomForestRegressor
            RandomForest estimator, used to predict fire cause ratios.
        xlabels : list
            Names (str) of the independent variables required by the estimator.
        modis_ba : pd.DataFrame, optional
            Dataframe with MODIS BA data aggregated to NUTS3. The default is None.

        Returns
        -------
        self.unknown_nuts (column with unknown values (Ln_y) is filled)
        self.ratios_nuts (the ratios per fire source (human, lightning))
        
        when modis_ba provided:
        self.modis_ba (burned area per fire source)
        self.country_ratios (ratios of fire source per country)

        """
        
        # Create a copy of the class' dfs'
        df_nuts = self.unknown_ratios.copy()
        
        # Predict, using the estimator
        df_nuts[self.dependent_label] = estimator.predict(df_nuts[xlabels])
        
        # Convert predictions to actual ratio's
        self.ratios_nuts = pd.DataFrame(df_nuts["NUTS_ID"].copy())
        self.ratios_nuts["RATIO_Human"], self.ratios_nuts["RATIO_Lightning"] = self.ComputeRatio(df_nuts[self.dependent_label])
        
        # Update the df's
        self.unknown_ratios = df_nuts
        
        if re.search("._N", self.dependent.iloc[:,-1].name) and modis_ba:
            print("Burned Area cannot be calculated from Fire Incidence!")
        elif re.search("._BA", self.dependent.iloc[:,-1].name): 
            self.modis_ba = self.__ComputeBA__(modis_ba)
            self.country_ratios = self.__RatiosPerCountry__(modis_ba)
        
        return  
    
    
    def ComputeRatio(self, y_hat):
        """
        Compute the ratio of fire occurence, using modeled output.
        
        Parameters
        ----------
        y_hat : np.float, or np.array etc.
            The model output for the predctor variable: ln(N_RATIO_Human / N_RATIO_Lightning)
    
        Returns
        -------
        human_ratio : np.float, or np.array etc.
            The ratio of human induced fires.
        lightning_ratio : np.float, or np.array etc.
            The ratio of lightning induced fires.
    
        """
        ey = np.exp(y_hat)
        
        human_ratio = ey / (1 + ey)
        lightning_ratio = 1 - human_ratio
        
        return human_ratio, lightning_ratio


    def ExportToCSV(self, out_path, geo_name = "NUTS_ID", extra_cols = None):
        """ Export predicted data to csv file """
        
        # First check if predictions have been made, else exporting would be useless
        if not hasattr(self, "ratios_nuts"):
            print("Please first predict for the unknown regions, before exporting!")
            return 
        
        # Check if we are eporting fire incidence or burned area
        if re.search("._N", self.dependent.iloc[:,-1].name):
            tag = "N_RATIO"
        else: 
            tag = "BA_RATIO"
        
        # Create a DataFrame containing the observations and predictions for all regions
        if geo_name == "NUTS_ID":
            df = pd.DataFrame(self.raw[[geo_name, f"{tag}_Human", f"{tag}_Lightning"]].copy()) # Get the known data
            df = pd.merge(df, self.ratios_nuts, on = geo_name, how="outer") # Merge with the predictions
            df = df.rename(columns = {f"{tag}_Human" : "Obs_R_Human",           # Rename for clarification
                                      f"{tag}_Lightning" : "Obs_R_Lightning", 
                                      "RATIO_Human" : "Pr_R_Human", 
                                      "RATIO_Lightning" : "Pr_R_Lightning"})
            ln_y = pd.concat([self.unknown_ratios[["NUTS_ID", f"{self.dependent_label}"]], self.dependent])
            df = pd.merge(ln_y, df, on = geo_name, how="outer") # Also add the dependent variable to the df
        elif geo_name != "NUTS_ID" and tag == "BA_RATIO" and hasattr(self, "country_ratios"):
            df = self.country_ratios
            extra_cols = None
        
        
        # If desired, add extra columns to the output, such as uncertainty estimate
        if type(extra_cols) != type(None): 
            df = pd.merge(df, extra_cols, on = geo_name, how="outer")
        
        # If we are dealing with burned area, and modis_ba is known, 
        # also add a column ba differentiated per source
        if geo_name == "NUTS_ID" and tag == "BA_RATIO" and hasattr(self, "modis_ba"):
            df = pd.merge(df, self.modis_ba, on = geo_name, how="outer")
            
        # The actual export
        df.to_csv(out_path, sep = ';')

        return
    
    
    def DetermineCorrelations(self, df, ylabels, xlabels, confidence=0.95):
        """ 
        Generate Correlation Matrix and determine p-values 
        
        Returns: spearman_p, pearson_p, spearman_corr, pearson_corr
        """
        
        # Check if the input data is of correct type, else correct it
        if not type(ylabels) == list:
            ylabels = [ylabels]
        if not type(ylabels) == list:
            xlabels = [xlabels]
        
        matrix_labels = ylabels + xlabels # Labels to be used in the correlation Matrix
        
        # Initiate matrices
        spearman_corr = pd.DataFrame(float('nan'), index=matrix_labels, columns=matrix_labels)
        pearson_corr = pd.DataFrame(float('nan'), index=matrix_labels, columns=matrix_labels)
    
        spearman_p = pd.DataFrame(float('nan'), index=matrix_labels, columns=matrix_labels)
        pearson_p = pd.DataFrame(float('nan'), index=matrix_labels, columns=matrix_labels)
        
        # Fill the matrices with correlation coefficients and p-values
        for i, x in enumerate(spearman_corr.columns):
            for j, y in enumerate(spearman_corr.columns):
                # Determine the pearson and spearman coefficients
                corrs, p_val_s = spearmanr(a = df[x], b = df[y])
                corrp, p_val_p = pearsonr(x = df[x], y = df[y])
                            
                # Append these to the DataFrame
                spearman_corr[x][y] = corrs
                pearson_corr[x][y] = corrp
                
                # Also denote the p-values
                spearman_p[x][y] = p_val_s
                pearson_p[x][y] = p_val_p
                
        # Set all non-significant correlations to 'nan'
        spearman_corr[spearman_p > (1 - confidence)] = float('nan') # Remove non-significant correlations
        pearson_corr[pearson_p > (1 - confidence)] = float('nan') # Remove non-significant correlations
        
        return spearman_p, pearson_p, spearman_corr, pearson_corr


    def __ComputeBA__(self, modis_ba):
        """ Compute the Burned Area per NUTS3 region, differentiated per source (human or lightning) """
        # Get dependent variable
        df = self.dependent.copy().set_index('NUTS_ID')
        Ln_y = self.unknown_ratios[["NUTS_ID", self.dependent_label]].copy().set_index("NUTS_ID")
        Ln_y = pd.concat([df, Ln_y])
        #Ln_y = np.array(Ln_y.sort_index(axis=0))
        #Ln_y.fillna(df, inplace = True)
        
        # Get both predicted and known ratio's
        df = self.ratios_nuts[["NUTS_ID", "RATIO_Human"]].copy().set_index('NUTS_ID')
        ch = self.raw[["NUTS_ID", "BA_RATIO_Human"]].copy().set_index('NUTS_ID')
        ch = ch.rename(columns = {"BA_RATIO_Human" : "RATIO_Human"})
        ch.fillna(df, inplace = True)
        #ch = np.array(ch.sort_index(axis=0))
        
        arr = np.array(pd.merge(ch, Ln_y, on = "NUTS_ID", how="outer").sort_index(axis=0))
        #ch = ch.reset_index()
        
        modis_pixel_area = 111 * 0.004850803768137106758**2 # km2
        modis_ba = modis_ba.set_index("NUTS_ID")
        modis_ba = modis_ba.sort_index(axis = 0)
        idx = modis_ba.index
        modis_ba = np.squeeze(np.array(modis_ba)) * modis_pixel_area
        
        ba_human = modis_ba * (1 - (arr[:,0] / np.exp(arr[:,1])))
        ba_lightning = modis_ba - ba_human
        
        df = pd.DataFrame({"NUTS_ID" : idx, "BA_Human_km2" : ba_human, "BA_Lightning_km2" : ba_lightning})
        
        return df
    

    def __DetermineCountryRatios__(self):
        """ 
        This function generates a pandas DataFrame with the aggregated weighted mean of all 
        independent variables per country. This data can then be fed to a RandomForestRegressor to estimate 
        the ln(human_ratio / lightning_ratio).

        NOTE this function is DEPRACATED, as the fire ratios per country are now determined using MODIS BA data:
            ratio_human(country) = sum(BA_human(nuts3)) / sum(mean_annual_modis_BA(country))
            
        """
        
        x, labels = self.__IndependentVariablesDF__(self.raw["NUTS_ID"]) # Get value of all NUTS zones and their vals.
        
        cntr_df = pd.DataFrame(columns = self.independent_labels) # Initiate df to append data to
        
        for cntr in self.raw["CNTR_CODE"].unique():
            cntr_y = self.raw.loc[self.raw['CNTR_CODE'] == cntr] # Select all NUTS regions (+area) belonging to a country
            cntr_x = x.loc[x["NUTS_ID"].str.contains(cntr)] # Select x data belonging to the country
            
            xy_df = pd.merge(cntr_y[["NUTS_ID", "NUTS_Area"]], cntr_x, on = "NUTS_ID")
            
            # Calculate weighted mean (area-wise) of all NUTS regions
            total_area = np.sum(xy_df["NUTS_Area"]) # km2
            weighted_sum = xy_df.iloc[:,2:].multiply(xy_df.iloc[:,1], axis="index").sum(axis = 0)
            
            weighted_mean = (weighted_sum / total_area).rename(cntr)
            
            cntr_df = cntr_df.append(weighted_mean, ignore_index = False)
        
        # Now sort the df per country and add column for predictions to it
        cntr_df['CNTR_CODE'] = cntr_df.index
        df = cntr_df['CNTR_CODE'].copy().reset_index()
        df.pop('index')
        df[self.dependent_label] = float('nan')
        df = pd.merge(df, cntr_df, on = 'CNTR_CODE')
        
        return df
    
    
    def __IndependentVariablesDF__(self, geo_col):
        """ Summarize all independent variable data into one df """
        
        labels = []
        geo_name = geo_col.name
        
        for i, iv in enumerate(self.iv_list):
            geo_col = pd.merge(geo_col, iv.data, on=[geo_name]) # Append to the dependent variable dataframe
            geo_col.rename(columns = {iv.data.columns[-1] : iv.name}, inplace = True)      
            labels.append(iv.name)
            
        return geo_col, labels
    
    
    def __RatiosPerCountry__(self, modis_ba):
        print("Determining the fire cause ratio's per country, using MODIS BA data.")
        
        cntr_df = pd.DataFrame(columns = ["CNTR_CODE", "BA_RATIO_Human", "BA_RATIO_Lightning"])
        for cntr in self.raw["CNTR_CODE"].unique():
            # Select all NUTS regions belonging to a country
            ba_human_lightning = self.modis_ba[self.modis_ba['NUTS_ID'].str.startswith(cntr)]
            ba_modis = modis_ba[modis_ba['NUTS_ID'].str.startswith(cntr)].copy() # MODIS BA
            ba_modis["km2"] = (ba_modis.iloc[:,-1].copy() * 111 * 0.004850803768137106758**2) # km2
            
            ratio_human = np.divide(np.nansum(ba_human_lightning["BA_Human_km2"]), np.nansum(ba_modis["km2"]), 
                                    where = np.nansum(ba_modis["km2"]) != 0)
            ratio_lightning = 1 - ratio_human
            
            cntr_df = cntr_df.append({"CNTR_CODE": cntr, 
                                      "BA_RATIO_Human" : ratio_human, 
                                      "BA_RATIO_Lightning": ratio_lightning}, ignore_index=True)
        
        return cntr_df
    
    
    def __UnknownRatios__(self):
        """ Create df with independent variables sorted per NUTS region, for the NUTS regions that lack EFFIS fire data """
        
        if re.search("._N", self.dependent.iloc[:,-1].name):
            f = "N_RATIO_Human"
        else: 
            f = "BA_RATIO_Human"

        ylabels = self.raw[self.raw[f].isna()]
        
        x, labels = self.__IndependentVariablesDF__(self.raw["NUTS_ID"])
        
        df_nuts = pd.merge(ylabels[["NUTS_ID", "NUTS_Area", f]], x, on="NUTS_ID")
        df_nuts = df_nuts.rename(columns = {f : self.dependent_label})
        
        return df_nuts
    
    