"""
Copyright (C) Microsoft Corporation. All rights reserved.​
 ​
Microsoft Corporation (“Microsoft”) grants you a nonexclusive, perpetual,
royalty-free right to use, copy, and modify the software code provided by us
("Software Code"). You may not sublicense the Software Code or any use of it
(except to your affiliates and to vendors to perform work on your behalf)
through distribution, network access, service agreement, lease, rental, or
otherwise. This license does not purport to express any claim of ownership over
data you may have shared with Microsoft in the creation of the Software Code.
Unless applicable law gives you more rights, Microsoft reserves all other
rights not expressly granted herein, whether by implication, estoppel or
otherwise. ​
 ​
THE SOFTWARE CODE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
MICROSOFT OR ITS LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THE SOFTWARE CODE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
import json

from sklearn.externals import joblib
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn_pandas import DataFrameMapper
import os
import pandas as pd

from azureml.core import Run


parser = argparse.ArgumentParser("train")
parser.add_argument(
    "--config_suffix", type=str, help="Datetime suffix for json config files"
)
parser.add_argument(
    "--json_config",
    type=str,
    help="Directory to write all the intermediate json configs",
)
parser.add_argument(
    "--model_name",
    type=str,
    help="Name of the Model",
    default="attrition_model.pkl",
)

args = parser.parse_args()

print("Argument 1: %s" % args.config_suffix)
print("Argument 2: %s" % args.json_config)

model_name = args.model_name

if not (args.json_config is None):
    os.makedirs(args.json_config, exist_ok=True)
    print("%s created" % args.json_config)


run = Run.get_context()
exp = run.experiment
ws = run.experiment.workspace




os.makedirs('./outputs', exist_ok=True)

# Get the IBM employee attrition dataset
attritionData = pd.read_csv('./Data/WA_Fn-UseC_-HR-Employee-Attrition.csv')

# Dropping Employee count as all values are 1 and hence attrition is independent of this feature
attritionData = attritionData.drop(['EmployeeCount'], axis=1)
# Dropping Employee Number since it is merely an identifier
attritionData = attritionData.drop(['EmployeeNumber'], axis=1)
attritionData = attritionData.drop(['Over18'], axis=1)
# Since all values are 80
attritionData = attritionData.drop(['StandardHours'], axis=1)

# Converting target variables from string to numerical values
target_map = {'Yes': 1, 'No': 0}
attritionData["Attrition_numerical"] = attritionData["Attrition"].apply(lambda x: target_map[x])
target = attritionData["Attrition_numerical"]

attritionXData = attritionData.drop(['Attrition_numerical', 'Attrition'], axis=1)

# Creating dummy columns for each categorical feature
categorical = []
for col, value in attritionXData.iteritems():
    if value.dtype == 'object':
        categorical.append(col)

# Store the numerical columns in a list numerical
numerical = attritionXData.columns.difference(categorical)

numeric_transformations = [([f], Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])) for f in numerical]
    
categorical_transformations = [([f], OneHotEncoder(handle_unknown='ignore', sparse=False)) for f in categorical]

transformations = numeric_transformations + categorical_transformations

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
clf = Pipeline(steps=[('preprocessor', DataFrameMapper(transformations)),
                      ('classifier', LogisticRegression(solver='lbfgs'))])

# Split data into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(attritionXData, 
                                                    target, 
                                                    test_size = 0.2,
                                                    random_state=0,
                                                    stratify=target)

# write x_text out as a pickle file for later visualization
x_test_pkl = 'x_test.pkl'
with open(x_test_pkl, 'wb') as file:
    joblib.dump(value=x_test, filename=os.path.join('./outputs/', x_test_pkl))

# preprocess the data and fit the classification model
clf.fit(x_train, y_train)
model = clf.steps[-1][1]

model_file_name = 'log_reg.pkl'

# save model in the outputs folder so it automatically get uploaded
with open(model_file_name, 'wb') as file:
    joblib.dump(value=clf, filename=os.path.join('./outputs/',
                                                 model_file_name))

# Explain predictions on your local machine
# tabular_explainer = TabularExplainer(model, initialization_examples=x_train, features=attritionXData.columns, classes=["Not leaving", "leaving"], transformations=transformations)

# Explain overall model predictions (global explanation)
# Passing in test dataset for evaluation examples - note it must be a representative sample of the original data
# x_train can be passed as well, but with more examples explanations it will
# take longer although they may be more accurate
# global_explanation = tabular_explainer.explain_global(x_test)

# Uploading model explanation data for storage or visualization in webUX
# The explanation can then be downloaded on any compute
# comment = 'Global explanation on classification model trained on IBM employee attrition dataset'

# ScoringExplainer
# scoring_explainer = tabular_explainer.create_scoring_explainer(x_test)
# Pickle scoring explainer locally
# scoring_explainer_path = scoring_explainer.save('scoring_explainer_deploy')

# client = ExplanationClient.from_run(run)
# client.upload_model_explanation(global_explanation, comment=comment)
# upload x_test, pickled above
run.upload_file('x_test_ibm.pkl', os.path.join('./outputs/', x_test_pkl))

# Register original model
run.upload_file('original_model.pkl', os.path.join('./outputs/', model_file_name))








# upload the model file explicitly into artifacts
run.upload_file(name="./outputs/" + model_name, path_or_stream=model_name)
print("Uploaded the model {} to experiment {}".format(
    model_name, run.experiment.name))
dirpath = os.getcwd()
print(dirpath)
print("Following files are uploaded ")
print(run.get_file_names())

run_id = {}
run_id["run_id"] = run.id
run_id["experiment_name"] = run.experiment.name
filename = "run_id_{}.json".format(args.config_suffix)
output_path = os.path.join(args.json_config, filename)
with open(output_path, "w") as outfile:
    json.dump(run_id, outfile)

run.complete()
