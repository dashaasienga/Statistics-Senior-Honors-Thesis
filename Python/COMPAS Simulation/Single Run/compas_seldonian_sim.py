#! /usr/bin/env python

# import necessary libraries
import numpy as np
import os

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

# point to the data file
#f_orig = "/home/dasienga24/Statistics-Senior-Honors-Thesis/Data Sets/COMPAS/compas_sim.csv"
f_orig = sys.argv[1]

# save final dataframe
output_path_data="/home/dasienga24/Statistics-Senior-Honors-Thesis/Data Sets/COMPAS/Simulation Single Run/compas_seldonian_sim.csv"

# save metadata json file
output_path_metadata="/home/dasienga24/Statistics-Senior-Honors-Thesis/Data Sets/COMPAS/Simulation Single Run/compas_seldonian_sim.json"


columns_orig = ["race","prior_offense","age", "is_recid"]

df = pd.read_csv(f_orig, header=0, names=columns_orig)




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

# change names of columns 
outdf.rename(columns={'c__race_African-American':'Black', 'c__race_Caucasian':'White', 'n__age':'age'}, inplace=True)

# re-index in order to concatenate columns
prior_offense = df["prior_offense"]
y.index = range(0, len(y))
prior_offense.index = range(0, len(prior_offense))

# add label column and `prior_offense` into final dataframe
outdf['prior_offense'] = prior_offense
outdf['is_recid'] = y



outdf.to_csv(output_path_data,index=False,header=False)


metadata_dict = {
  "regime":"supervised_learning",
  "sub_regime":"classification",
  "all_col_names":list(outdf.columns),
  "label_col_names":"is_recid",
  "sensitive_col_names":["Black", "White"]
}
    
with open(output_path_metadata,'w') as outfile:
    json.dump(metadata_dict,outfile,indent=2)
