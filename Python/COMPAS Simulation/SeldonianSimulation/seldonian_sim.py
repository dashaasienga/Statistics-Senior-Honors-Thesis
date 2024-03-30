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


----## SETTING IMPORTANT VALUES ##----

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


----## DATA PRE-PROCESSING ##----

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
    

----## FITTING A SELDONIAN ALGORITHM ##----
    
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

