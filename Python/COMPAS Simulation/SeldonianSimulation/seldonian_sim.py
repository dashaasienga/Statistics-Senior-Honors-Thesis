#! /usr/bin/env python

# import necessary libraries
import numpy as np
import os
import re

import pandas as pd
import json

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from seldonian.utils.io_utils import save_json
from seldonian.parse_tree.parse_tree import (ParseTree,
    make_parse_trees_from_constraints)
from seldonian.dataset import DataSetLoader
from seldonian.utils.io_utils import (load_json,save_pickle)
from seldonian.spec import SupervisedSpec
from seldonian.models.models import (
    BinaryLogisticRegressionModel as LogisticRegressionModel) 
from seldonian.models import objectives

import sys


##---- SETTING IMPORTANT VALUES ----##

# point to the data file
#f_orig = sys.argv[1]

# for testing
f_orig = "/home/dasienga24/Statistics-Senior-Honors-Thesis/Data Sets/SimulationData/sim_500_1.csv"

# extract size and dataset_id from simulation file using regular expressions
match = re.search(r'sim_(\d+)_(\d+).csv', f_orig)

size = int(match.group(1))
dataset_id = int(match.group(2))
    
# path to save Seldonian clean data temporarily
output_path_data="/home/dasienga24/Statistics-Senior-Honors-Thesis/Data Sets/Seldonian Clean Sim Data/temp.csv"

# path to save metadata json file temporarily
output_path_metadata="/home/dasienga24/Statistics-Senior-Honors-Thesis/Data Sets/Seldonian Clean Sim Data/temp.json"

# select specific columns of interest
columns_orig = ["race","prior_offense","age", "is_recid"]

# read in the simulation data set
df = pd.read_csv(f_orig, header=0, names=columns_orig)


##---- DATA PRE-PROCESSING ----##

# select inputs to be transformed 
X = df.drop(columns=["is_recid"])
y = df["is_recid"]

# one hot encode race, scale age using standard scaler 
ct = ColumnTransformer([('c',OneHotEncoder(),['race']), ('n',StandardScaler(),['age'])])
    
# apply transformation
X_transformed = ct.fit_transform(X)
    
# get names after one-hot encoding
output_columns = ct.get_feature_names_out(ct.feature_names_in_)
    
# make an output dataframe to save transformed X and y
outdf = pd.DataFrame(X_transformed,columns=output_columns)

# rename columns 
outdf.rename(columns={'c__race_African-American':'Black', 'c__race_Caucasian':'White', 'n__age':'age'}, inplace=True)

# re-index in order to properly concatenate columns
prior_offense = df["prior_offense"]
y.index = range(0, len(y))
prior_offense.index = range(0, len(prior_offense))

# add label column and `prior_offense` back into final dataframe
outdf['prior_offense'] = prior_offense
outdf['is_recid'] = y

# save clean data to temporary location defined above
outdf.to_csv(output_path_data,index=False,header=False)

# save json file to temporary location defined above
metadata_dict = {
  "regime":"supervised_learning",
  "sub_regime":"classification",
  "all_col_names":list(outdf.columns),
  "label_col_names":"is_recid",
  "sensitive_col_names":["Black", "White"]
}
    
with open(output_path_metadata,'w') as outfile:
    json.dump(metadata_dict,outfile,indent=2)
    

##---- CREATING THE SELDONIAN SPEC FILES ----##
    
import autograd.numpy as np

# path to temporarily store the seldonian spec files 
save_dir = "/home/dasienga24/Statistics-Senior-Honors-Thesis/Python/COMPAS Simulation/SeldonianSimulation/"
os.makedirs(save_dir,exist_ok=True)

# load Seldonian dataset from data and metadata file
regime='supervised_learning'
sub_regime='classification'

loader = DataSetLoader(regime=regime)

dataset = loader.load_supervised_dataset(
  filename=output_path_data,
  metadata_filename=output_path_metadata,
  file_type='csv')
  
sensitive_col_names = dataset.meta.sensitive_col_names

# use logistic regression model as a starting point
model = LogisticRegressionModel()
    
# set the primary objective to be log loss
primary_objective = objectives.binary_logistic_loss

from seldonian.spec import createSupervisedSpec

# define behavioral constraints (epsilon = 0.2)
epsilon = 0.2
constraint_name = "equalized_odds"
if constraint_name == "equalized_odds":
  constraint_strs = [f'abs((FNR | [Black]) - (FNR | [White])) + abs((FPR | [Black]) - (FPR | [White])) <= {epsilon}'] 
deltas = [0.05]

# create spec file
save_dir = "/home/dasienga24/Statistics-Senior-Honors-Thesis/Python/COMPAS Simulation/SeldonianSimulation/temp_equalized_odds_0.2"

os.makedirs(save_dir, exist_ok=True) #create folder

createSupervisedSpec(
            dataset=dataset,
            metadata_pth=output_path_metadata,
            constraint_strs=constraint_strs,
            deltas=deltas,
            save_dir=save_dir,
            save=True,
            verbose=False)
            
#---------------------------------------------#

# define behavioral constraints (epsilon = 0.1)
epsilon = 0.1
constraint_name = "equalized_odds"
if constraint_name == "equalized_odds":
  constraint_strs = [f'abs((FNR | [Black]) - (FNR | [White])) + abs((FPR | [Black]) - (FPR | [White])) <= {epsilon}'] 
deltas = [0.05]

# create spec file
save_dir = "/home/dasienga24/Statistics-Senior-Honors-Thesis/Python/COMPAS Simulation/SeldonianSimulation/temp_equalized_odds_0.1"

os.makedirs(save_dir, exist_ok=True) #create folder

createSupervisedSpec(
            dataset=dataset,
            metadata_pth=output_path_metadata,
            constraint_strs=constraint_strs,
            deltas=deltas,
            save_dir=save_dir,
            save=True,
            verbose=False)
            
#-----------------------------------------------#
            
# define behavioral constraints (epsilon = 0.05)
epsilon = 0.05
constraint_name = "equalized_odds"
if constraint_name == "equalized_odds":
  constraint_strs = [f'abs((FNR | [Black]) - (FNR | [White])) + abs((FPR | [Black]) - (FPR | [White])) <= {epsilon}'] 
deltas = [0.05]

# create spec file
save_dir = "/home/dasienga24/Statistics-Senior-Honors-Thesis/Python/COMPAS Simulation/SeldonianSimulation/temp_equalized_odds_0.05"

os.makedirs(save_dir, exist_ok=True) #create folder

createSupervisedSpec(
            dataset=dataset,
            metadata_pth=output_path_metadata,
            constraint_strs=constraint_strs,
            deltas=deltas,
            save_dir=save_dir,
            save=True,
            verbose=False)
            
#----------------------------------------------#

# define behavioral constraints (epsilon = 0.01)
epsilon = 0.01
constraint_name = "equalized_odds"
if constraint_name == "equalized_odds":
  constraint_strs = [f'abs((FNR | [Black]) - (FNR | [White])) + abs((FPR | [Black]) - (FPR | [White])) <= {epsilon}'] 
deltas = [0.05]

# create spec file
save_dir = "/home/dasienga24/Statistics-Senior-Honors-Thesis/Python/COMPAS Simulation/SeldonianSimulation/temp_equalized_odds_0.01"

os.makedirs(save_dir, exist_ok=True) #create folder

createSupervisedSpec(
            dataset=dataset,
            metadata_pth=output_path_metadata,
            constraint_strs=constraint_strs,
            deltas=deltas,
            save_dir=save_dir,
            save=True,
            verbose=False)


##---- RUNNING THE SELDONIAN ENGINE ----##

from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.utils.io_utils import load_pickle

# load the spec file (epsilon = 0.2)
specfile = '/home/dasienga24/Statistics-Senior-Honors-Thesis/Python/COMPAS Simulation/SeldonianSimulation/temp_equalized_odds_0.2/spec.pkl'
spec = load_pickle(specfile)
SA_02 = SeldonianAlgorithm(spec)

#-----------------------------------#

# load the spec file (epsilon = 0.1)
specfile = '/home/dasienga24/Statistics-Senior-Honors-Thesis/Python/COMPAS Simulation/SeldonianSimulation/temp_equalized_odds_0.1/spec.pkl'
spec = load_pickle(specfile)
SA_01 = SeldonianAlgorithm(spec)

#-----------------------------------#

# load the spec file (epsilon = 0.05)
specfile = '/home/dasienga24/Statistics-Senior-Honors-Thesis/Python/COMPAS Simulation/SeldonianSimulation/temp_equalized_odds_0.05/spec.pkl'
spec = load_pickle(specfile)
SA_005 = SeldonianAlgorithm(spec)

#-----------------------------------#

# load the spec file (epsilon = 0.01)
specfile = '/home/dasienga24/Statistics-Senior-Honors-Thesis/Python/COMPAS Simulation/SeldonianSimulation/temp_equalized_odds_0.01/spec.pkl'
spec = load_pickle(specfile)
SA_001 = SeldonianAlgorithm(spec)


##---- SAVING METRICS ----##

## CONVERGENCE
passed_safety_02, solution_02 = SA_02.run(write_cs_logfile=True)

passed_safety_01, solution_01 = SA_01.run(write_cs_logfile=True)

passed_safety_005, solution_005 = SA_005.run(write_cs_logfile=True)

passed_safety_001, solution_001 = SA_001.run(write_cs_logfile=True)

## ACCURACY 

# separate the predictor variables from the sensitive variable and the response variable
X_outdf = outdf.drop(columns = ['is_recid', 'Black', 'White'])
X_sens = outdf[['Black', 'White']]
y_outdf = outdf['is_recid']


#----------------------------------#
# get the solution & store coefficients (epsilon = 0.2)
coefficients = SA_02.cs_result["candidate_solution"]

# get the intercept
intercept = coefficients[0]

# compute the predictive values
linear_combination = np.dot(X_outdf, coefficients[1:]) + intercept
pred_probs_02 = 1 / (1 + np.exp(-linear_combination))

#----------------------------------#
# get the solution & store coefficients (epsilon = 0.1)
coefficients = SA_01.cs_result["candidate_solution"]

# get the intercept
intercept = coefficients[0]

# compute the predictive values
linear_combination = np.dot(X_outdf, coefficients[1:]) + intercept
pred_probs_01 = 1 / (1 + np.exp(-linear_combination))

#----------------------------------#
# get the solution & store coefficients (epsilon = 0.05)
coefficients = SA_005.cs_result["candidate_solution"]

# get the intercept
intercept = coefficients[0]

# compute the predictive values
linear_combination = np.dot(X_outdf, coefficients[1:]) + intercept
pred_probs_005 = 1 / (1 + np.exp(-linear_combination))

#----------------------------------#
# get the solution & store coefficients (epsilon = 0.01)
coefficients = SA_001.cs_result["candidate_solution"]

# get the intercept
intercept = coefficients[0]

# compute the predictive values
linear_combination = np.dot(X_outdf, coefficients[1:]) + intercept
pred_probs_001 = 1 / (1 + np.exp(-linear_combination))

#----------------------------------#
# store results
seldonian_results = pd.DataFrame({'is_recid': y_outdf, 'pred_0.2': pred_probs_02, 'pred_0.1': pred_probs_01, 'pred_0.05': pred_probs_005, 'pred_0.01': pred_probs_001})
seldonian_results = pd.concat([X_outdf, X_sens, seldonian_results], axis = 1)

# define threshold
threshold = 0.5

# create risk columns
risk_02 = np.where(pred_probs_02 >= threshold, 1, 0)
risk_01 = np.where(pred_probs_01 >= threshold, 1, 0)
risk_005 = np.where(pred_probs_005 >= threshold, 1, 0)
risk_001 = np.where(pred_probs_001 >= threshold, 1, 0)

# add risk columns to dataframe
seldonian_results['risk_0.2'] = risk_02
seldonian_results['risk_0.1'] = risk_01
seldonian_results['risk_0.05'] = risk_005
seldonian_results['risk_0.01'] = risk_001

#----------------------------------#
# compute accuracy
sa_02_accuracy = (seldonian_results['risk_0.2'] == seldonian_results['is_recid']).sum() / len(seldonian_results)

sa_01_accuracy = (seldonian_results['risk_0.1'] == seldonian_results['is_recid']).sum() / len(seldonian_results)

sa_005_accuracy = (seldonian_results['risk_0.05'] == seldonian_results['is_recid']).sum() / len(seldonian_results)

sa_001_accuracy = (seldonian_results['risk_0.01'] == seldonian_results['is_recid']).sum() / len(seldonian_results)

## DISCRIMINATION

##---- SYNTHESIZING RESULTS ----#

##---- SAVING RESULTS ----#
