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
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV

class RandomForest:

    def __init__(self, x, y, labels, n_trees = 1000, random_state = 13, 
                 test_size = 0.3, scoring = 'explained_variance', param_dict=None):
        self.x = np.squeeze(np.array(x))
        self.y = np.squeeze(np.array(y))
        self.random_state = random_state
        self.test_size = test_size
        self.n_trees = n_trees
        self.labels = labels
        self.scoring = scoring # Scoring Method to optimize
        self.__paramdict__ = param_dict
        
        # Split Dataset into training and testing
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.x, self.y, test_size = test_size, \
                             random_state = random_state)
        
        if not param_dict:
            self.DefaultForest = self.InstantiateForest()
        else:
            self.DefaultForest = self.InstantiateForestWithParams(self.__paramdict__)
        
        return


    def InstantiateForest(self):
        print('Instantiating Random Forest with default parameters: \n')
        
        # Instantiate model with <n_trees> decision trees
        rf = RandomForestRegressor(n_estimators = self.n_trees, random_state = self.random_state)
        rf.fit(self.x_train, self.y_train) # Train the model on training data
        
        # Check Out the performance of the model
        r_squared, errors, accuracy = self.Evaluate(rf, print_output = True)
        self.DefaultForest_performance = {'R-squared':r_squared, 
                            'Mean Average Error (%)':np.nanmean(errors)*100,
                            'Accuracy (%)':accuracy}
        
        # Now also get Feature Importances
        self.DefaultForest_Importances = self.RelativeImportance(rf)
        
        return rf


    def InstantiateForestWithParams(self, param_dict):
        print('Instantiating Random Forest with specified parameters: \n')
        
        # Instantiate model with <n_trees> decision trees
        rf = RandomForestRegressor(n_estimators = param_dict["n_estimators"],
                                   bootstrap = param_dict["bootstrap"],
                                   max_depth = param_dict['max_depth'],
                                   max_features = param_dict['max_features'],
                                   min_samples_leaf = param_dict["min_samples_leaf"],
                                   min_samples_split = param_dict["min_samples_split"],
                                   random_state = self.random_state)
        rf.fit(self.x_train, self.y_train) # Train the model on training data
        
        # Check Out the performance of the model
        r_squared, errors, accuracy = self.Evaluate(rf, print_output = True)
        self.DefaultForest_performance = {'R-squared':r_squared, 
                            'Mean Average Error (%)':np.nanmean(errors)*100,
                            'Accuracy (%)':accuracy}
        
        # Now also get Feature Importances
        self.DefaultForest_Importances = self.RelativeImportance(rf)
        
        return rf
    
    
    def RandomizedGridSearch(self, n_param_samples = 100):
        print('Randomized Grid Search CV: \n')
        
        param_grid = self.__BaseParamGrid__() # Initialise Base Param Grid
        forest = RandomForestRegressor(random_state = self.random_state) # Initialize forest regressor
        
        # Initialize Randomized Search CV
        randomized = RandomizedSearchCV(forest, param_grid, 
                                        n_iter=n_param_samples, # Amount of parameter configurations that are sampled
                                        cv=3, # cross-validation splitting strategy.
                                        scoring = self.scoring, # Score to optimize
                                        n_jobs=-1, verbose=2) 
        
        # Fit the Randomized Search CV 
        randomized.fit(self.x_train, self.y_train)
        
        # Check Out the performance of the Random Grid Search
        r_squared, errors, accuracy = self.Evaluate(randomized, print_output = True)
        self.RandomGridSearch_Performance = {'R-squared':r_squared, 
                                            'Mean Average Error (%)':np.nanmean(errors)*100,
                                            'Accuracy (%)':accuracy}
        
        # Now also get Feature Importances
        self.RandomGridSearch_Importances = self.RelativeImportance(randomized.best_estimator_)

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
        r_squared, errors, accuracy = self.Evaluate(self.GridSearch_, print_output = True)
        self.GridSearch_Performance = {'R-squared':r_squared, 
                                       'Mean Average Error (%)':np.nanmean(errors)*100,
                                       'Accuracy (%)':accuracy}
        
        # Now also get Feature Importances
        self.GridSearch_Importances = self.RelativeImportance(self.GridSearch_.best_estimator_)

        # Save Some Results in the Random Forest Object
        self.GridSearch_Params = self.GridSearch_.best_params_
        self.GridSearch_Estimator = self.GridSearch_.best_estimator_
        
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
        predictions = model.predict(self.x_test)
        errors = abs(predictions - self.y_test)
        mape = 100 * np.mean(errors / self.y_test)
        accuracy = 100 - mape
        
        if print_output:
            print('Model Performance:')
            print('R-Squared test data:', np.round(r_squared, 2))
            print('Mean Absolute Error: {:0.4f} %.'.format(np.mean(errors) * 100))
            print('Accuracy = {:0.2f}%. \n'.format(accuracy))
    
        return r_squared, errors, accuracy
    
    
    def RelativeImportance(self, model):
        importances = list(model.feature_importances_)
        zip_iterator = zip(self.labels, importances)
        return dict(zip_iterator)


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
            max_depth = param_grid["max_depth"]
    
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
