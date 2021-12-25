# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 14:22:18 2021

@author: Jasper Dijkstra

The class in this script performs a Random Forest Analysis:

By calling the RandomForest class:
    - A Random forest analysis with default parameters is fitted
    - The performance and predictor variable importance are assessed.
    - The fit might be improved, by using Hyperparameter tuning:
        - First call the .RandomizedGridSearch() method, 
          which takes the amount of samples to draw as an argument (n_param_samples)
        - Secondly, call the .GridSearch method.

Additional information
- Info in Random Forests: https://www.analyticsvidhya.com/blog/2020/03/beginners-guide-random-forest-hyperparameter-tuning/
- Different options for the 'scoring' parameter: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring
- Possibly useful ways to visualise results: https://github.com/mahesh147/Random-Forest-Classifier/blob/master/random_forest_classifier.py


"""
import pandas as pd
import numpy as np

# Sklearn Imports
from sklearn.ensemble import RandomForestRegressor
#from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import explained_variance_score


class UncertaintyTest:
    
    def __init__(self, x_train, x_test, y_train, y_test, idx_train, idx_test, estimator, performance, importances):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train 
        self.y_test = y_test
        self.idx_train = idx_train 
        self.idx_test = idx_test

        self.estimator = estimator
        self.performance = performance
        self.importances = importances
        
        return


class ForestMethods:
    
    def __init__(self, x_test, x_train, y_test, y_train, labels, model):
        self.x_test = x_test
        self.x_train = x_train
        self.y_test = y_test
        self.y_train = y_train
        self.labels = labels
        self.model = model
        return
        
    
    def Evaluate(self, model, print_output=True):
        """
        Assess the performance of the model in terms of: r-squared, 
        average error & accuracy
        
        returns: dict
        """
        
        # R-Squared
        r_squared = model.score(X = self.x_test, y = self.y_test)
        
        # Errors in predction
        predictions = model.predict(self.x_test) # Predictions
        y_hat = np.exp(predictions) / (1 + np.exp(predictions)) # Convert predictions to ratios
        y = np.exp(self.y_test) / (1 + np.exp(self.y_test)) # Convert y_test to ratios
        mae = np.nansum(abs(y_hat - y)) / len(y)
        variance = explained_variance_score(self.y_test, predictions)
        
        if print_output:
            print('Model Performance:')
            print('R-Squared test data:', np.round(r_squared, 2))
            print('Mean Absolute Error: {:0.4f}'.format(mae))
            print('Explained Variance: {:0.2f} \n'.format(variance))
    
        return r_squared, mae, variance
    
    
    def RelativeImportance(self, model):
        # Use Permutation importances
        # https://scikit-learn.org/stable/modules/permutation_importance.html#permutation-importance
        # importances = permutation_importance(model, self.x_test, self.y_test, n_repeats=30)#, random_state=0)
        
        # Use Imupurity Importance
        importances = list(model.feature_importances_)
        
        zip_iterator = zip(self.labels, importances)
        return dict(zip_iterator)


class RandomForest:

    def __init__(self, x, y, labels, n_trees = 1000, random_state = 13, random_tests = 100,
                 test_size = 0.3, scoring = 'explained_variance', param_dict=None, 
                 estimate_uncertainty = True, predict_arr=None):

        self.x = np.squeeze(np.array(x))
        self.y = np.squeeze(np.array(y))
        indices = np.arange(len(y))
        #self.random_state = random_state
        self.test_size = test_size
        self.n_trees = n_trees
        self.labels = labels
        self.scoring = scoring # Scoring Method to optimize
        self.__paramdict__ = param_dict
        
        if not param_dict:
                self.x_train, self.x_test, self.y_train, self.y_test, self.idx_train, self.idx_test = \
                                            train_test_split(self.x, self.y, indices, test_size = test_size)
                self.DefaultForest = self.InstantiateForest()
        elif param_dict and estimate_uncertainty:
            if predict_arr is None:
                predict_arr = self.x
            
            # We will iteratre over several combinations of train/test data:
            estimatorlist = [] # Initiate list to append data of each iteration to
            performance_list = [] # Initiate list to append performace of iteration to
            results_matrix = np.zeros((len(predict_arr), random_tests)) # Initiate matrix to append predicted results for each iteration to
            results_matrix[results_matrix == 0] = np.nan # Set all values in matrix to 'nodata'
            
            test_results_matrix = np.zeros((len(self.x), random_tests)) # Initiate matrix with all test results
            test_results_matrix[test_results_matrix == 0] = np.nan # Set all values in matrix to 'nodata'
            
            print(f"Determining the best Random Forest Estimator for {random_tests} different train/test combinations \n")
            
            for i in range(random_tests):
                
                # Split Dataset into training and testing
                self.x_train, self.x_test, self.y_train, self.y_test, self.idx_train, self.idx_test = \
                                            train_test_split(self.x, self.y, indices, test_size = test_size)
                
                # Fit a Random forest with specified parameters
                estimator, performance, importances = self.InstantiateForestWithParams(self.__paramdict__)
                
                # Performance metric to base result on
                performance_list.append(performance["R-squared"])
                
                # Predict for test set
                y_pred = estimator.predict(self.x_test) # Predict for the test set
                y_pred = np.exp(y_pred) / (1 + np.exp(y_pred)) # Convert ln(y) to human ratio's, to ease interpretation later
                test_results_matrix[self.idx_test, i] = y_pred # Append the results to the results matrix, at the correct idx
                
                # Predict for unknown regions
                y_pred = estimator.predict(predict_arr) # predict for the unknown ratios
                y_pred = np.exp(y_pred) / (1 + np.exp(y_pred)) # Convert ln(y) to human ratio's, to ease interpretation later
                results_matrix[:, i] = y_pred

                # Save results, so we can find them back later
                estimatorlist.append(UncertaintyTest(self.x_train, self.x_test, 
                                                     self.y_train, self.y_test,
                                                     self.idx_train, self.idx_test, 
                                                     estimator, performance, importances))
            
            # Determine at what iteration the result was best:
            best_i = np.where(np.array(performance_list) == np.max(np.array(performance_list)))[0][0]
            
            # Now set all values of the best as the default
            self.x_train = estimatorlist[best_i].x_train
            self.x_test = estimatorlist[best_i].x_test
            self.y_train = estimatorlist[best_i].y_train
            self.y_test = estimatorlist[best_i].y_test
            self.idx_train = estimatorlist[best_i].idx_train
            self.idx_test = estimatorlist[best_i].idx_test
            self.DefaultForest = estimatorlist[best_i].estimator
            self.DefaultForest_performance = estimatorlist[best_i].performance
            self.DefaultForest_Importances = estimatorlist[best_i].importances
            self.score_uncertainty = performance_list
            
            # Now we want to get an uncertainty estimate
            nans = ~np.isnan(results_matrix)
            self.n_unc_predictions = np.sum(nans, axis=1) # Count the number of valid points
            self.std_predictions = np.nanstd(results_matrix, axis=1) # Get an uncertainty (std) for each NUTS region

            # Now we want to get an uncertainty estimate
            nans = ~np.isnan(test_results_matrix)
            self.n_test_results = np.sum(nans, axis=1)
            self.average_test_results = np.nanmean(test_results_matrix, axis=1)
        
        else: # If we do not want to estimate the uncertainty, we can just use the first random train/test set
            self.x_train, self.x_test, self.y_train, self.y_test, self.idx_train, self.idx_test = \
                                        train_test_split(self.x, self.y, indices, test_size = test_size)
            self.DefaultForest = self.InstantiateForestWithParams(self.__paramdict__)
            
        return


    def InstantiateForest(self):
        print('Instantiating Random Forest with default parameters: \n')
        
        # Instantiate model with <n_trees> decision trees
        rf = RandomForestRegressor(n_estimators = self.n_trees)#, random_state = self.random_state)
        rf.fit(self.x_train, self.y_train) # Train the model on training data
        
        # Check Out the performance of the model
        forest_methods = ForestMethods(self.x_test, self.x_train, self.y_test, self.y_train, self.labels, rf) # First initiate methods
        r_squared, mae, exp_var = forest_methods.Evaluate(rf, print_output = True)
        self.DefaultForest_performance = {'R-squared':r_squared, 
                                          'Mean Absolute Error':mae,
                                          'Explained Variance':exp_var}
        
        # Now also get Feature Importances
        self.DefaultForest_Importances = forest_methods.RelativeImportance(rf)
        
        return rf


    def InstantiateForestWithParams(self, param_dict):
        print('Instantiating Random Forest with specified parameters: \n')
        
        # Instantiate model with <n_trees> decision trees
        rf = RandomForestRegressor(n_estimators = param_dict["n_estimators"],
                                   bootstrap = param_dict["bootstrap"],
                                   max_depth = param_dict['max_depth'],
                                   max_features = param_dict['max_features'],
                                   min_samples_leaf = param_dict["min_samples_leaf"],
                                   min_samples_split = param_dict["min_samples_split"])
        rf.fit(self.x_train, self.y_train) # Train the model on training data
        
        # Check Out the performance of the model
        forest_methods = ForestMethods(self.x_test, self.x_train, self.y_test, self.y_train, self.labels, rf) # First initiate methods
        r_squared, mae, exp_var = forest_methods.Evaluate(rf, print_output = True)
        performance = {'R-squared':r_squared, 
                       'Mean Absolute Error':mae,
                       'Explained Variance':exp_var}
        
        # Now also get Feature Importances
        importances = forest_methods.RelativeImportance(rf)
        
        return rf, performance, importances
    
    
    def RandomizedGridSearch(self, n_param_samples = 100):
        print('Randomized Grid Search CV: \n')
        
        param_grid = self.__BaseParamGrid__() # Initialise Base Param Grid
        forest = RandomForestRegressor()#random_state = self.random_state) # Initialize forest regressor
        
        # Initialize Randomized Search CV
        randomized = RandomizedSearchCV(forest, param_grid, 
                                        n_iter=n_param_samples, # Amount of parameter configurations that are sampled
                                        cv=3, # cross-validation splitting strategy.
                                        scoring = self.scoring, # Score to optimize
                                        n_jobs=-1, verbose=2) 
        
        # Fit the Randomized Search CV 
        randomized.fit(self.x_train, self.y_train)
        
        # Check Out the performance of the Random Grid Search
        forest_methods = ForestMethods(self.x_test, self.x_train, self.y_test, self.y_train, self.labels, randomized) # First initiate methods
        r_squared, mae, exp_var = forest_methods.Evaluate(randomized, print_output = True)
        self.RandomGridSearch_Performance = {'R-squared':r_squared, 
                                             'Mean Absolute Error': mae,
                                             'Explained Variance': exp_var}
         
        # Now also get Feature Importances
        self.RandomGridSearch_Importances = forest_methods.RelativeImportance(randomized.best_estimator_)

        # Save Some Results in the Random Forest Object
        self.RandomGridSearch_Params = randomized.best_params_
        self.RandomGridSearch_Estimator = randomized.best_estimator_
        self.RandomGridSearch = randomized

        return


    def GridSearch(self, init_params = 'random'):
        if not hasattr(self, "RandomGridSearch"):
            print("A Randomized Grid Search has to be performed prior to Grid Search! ... \n")
            print("Randomized Grid Search will be performed using default settings (n_param_samples = 100, scoring='explained_variance') \n")
            self.RandomizedGridSearch()
        
        print('Grid Search CV: \n')
        
        # Generate a Narrowed Grid...
        if init_params == 'random':
            param_grid = self.__NarrowedParamGrid__(self.RandomGridSearch_Params, n_estimators_step=100,
                                                    n_estimators_bigstep=False)
        else:
            param_grid = self.__NarrowedParamGrid__(self.GridSearch_Params, n_estimators_step=100,
                                                    n_estimators_bigstep=False)
            
        # Initialize forest regressor
        forest = RandomForestRegressor(random_state = self.random_state) # Initialize forest regressor
        
        # Execute Random Grid Search CV
        self.GridSearch_ = GridSearchCV(forest, param_grid, cv=3, 
                                       scoring = self.scoring, 
                                       n_jobs=-1, verbose=2)
        
        # Fit the Grid Search
        self.GridSearch_.fit(self.x_train, self.y_train)
        
        # Check Out the performance of the Random Grid Search
        forest_methods = ForestMethods(self.x_test, self.x_train, self.y_test, self.y_train, self.labels, self.GridSearch_) # First initiate methods
        r_squared, mae, exp_var = forest_methods.Evaluate(self.GridSearch_, print_output = True)
        self.GridSearch_Performance = {'R-squared':r_squared, 
                                       'Mean Absolute Error':mae,
                                       'Explained Variance':exp_var}
        
        # Now also get Feature Importances
        self.GridSearch_Importances = forest_methods.RelativeImportance(self.GridSearch_.best_estimator_)

        # Save Some Results in the Random Forest Object
        self.GridSearch_Params = self.GridSearch_.best_params_
        self.GridSearch_Estimator = self.GridSearch_.best_estimator_
        
        return


    def __BaseParamGrid__(self):
        
        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)] # Number of trees in random forest
        max_features = ['auto', 'sqrt', 'log2'] # Number of features to consider at every split
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)] # Maximum number of levels in tree
        max_depth.append(None)
        min_samples_split = [2, 5, 10] # Minimum number of samples required to split a node
        min_samples_leaf = [1, 2, 4] # Minimum number of samples required at each leaf node
        bootstrap = [True]#, False] # Method of selecting samples for training each tree
        
        # Build the random grid
        param_grid = {'n_estimators': n_estimators,
                        'max_features': max_features,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf,
                        'bootstrap': bootstrap}
        
        return param_grid

    
    def __NarrowedParamGrid__(self, param_grid, n_estimators_step=100, n_estimators_bigstep=True):
        
        # Method of selecting samples for training each tree
        bootstrap = [True]
        
        # Maximum number of levels in tree
        try:
            max_depth = [int(x) for x in np.linspace(start = param_grid["max_depth"] - 20, stop = param_grid["max_depth"] + 20, num = 5) if x > 0]
            #max_depth.append(None)
        except TypeError:
            max_depth = list(param_grid["max_depth"])
    
        # # Number of features to consider at every split
        max_features = [param_grid["max_features"]]

        # Minimum number of samples required at each leaf node
        min_samples_leaf = [int(x) for x in np.linspace(start = param_grid["min_samples_leaf"] - 1, stop = param_grid["min_samples_leaf"] + 1, num = 3) if x > 0]
        
        # Minimum number of samples required to split a node
        min_samples_split = [int(x) for x in np.linspace(start = param_grid["min_samples_split"] - 2, stop = param_grid["min_samples_split"] + 2, num = 3) if x > 0]
        
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = param_grid["n_estimators"] - (2 * n_estimators_step), stop = param_grid["n_estimators"] + n_estimators_step, num = 4) if x > 0]
        if n_estimators_bigstep:
            n_estimators.append(param_grid["n_estimators"] + (5 * n_estimators_step))
        
        # Putting all results in a new parameter grid
        narrowed_params = {
            'bootstrap': bootstrap,
            'max_depth': max_depth,
            'max_features': max_features,
            'min_samples_leaf': min_samples_leaf,
            'min_samples_split': min_samples_split,
            'n_estimators': n_estimators
        }
        return narrowed_params


