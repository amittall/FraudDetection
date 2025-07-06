import pandas as pd
from collections import Counter
from scipy import stats
import numpy as np 

df = pd.read_csv("/root/katacoda-scenarios/fraud-detection-data-prep/assets/fraud_detection_data.csv")

df['card_number'] = df['card_number'].astype(str)

print(df.head())

print(df.info())

print(df.describe())

print(df.columns)

print(Counter(df['fraud_flag']))

import ast 

df['merchant_state'] = df['merchant_state'].astype('category')
df['merchant_state_code'] = df['merchant_state'].cat.codes

df['merchant_city'] = df['merchant_city'].astype('category')
df['merchant_city_code'] = df['merchant_city'].cat.codes


df['card_type'] = df['card_type'].astype('category')
df['card_type_code'] = df['card_type'].cat.codes


df['cardholder_name'] = df['cardholder_name'].astype('category')
df['cardholder_name_code'] = df['cardholder_name'].cat.codes

number_of_items = [len(ast.literal_eval(x)) for x in list(df['items'])]

df['number_of_items'] = number_of_items

threshold = 3
z_scores = np.abs(stats.zscore(df['transaction_amount']))
df_no_outliers = df[(z_scores < threshold)]

print("CATEGORICAL VARIABLES HAVED BEEN ENCODED AND OUTLIERS HAVE BEEN REMOVED")


features  = ['merchant_state_code','merchant_city_code', 'card_type_code','cardholder_name_code',
             'transaction_amount', 'number_of_items']
target = 'fraud_flag'

X = df_no_outliers[features]

y = df_no_outliers[target]

X.to_csv("features.csv", index=False)

y.to_csv("targets.csv", index=False)

print("FEATURES AND TARGETS HAVE BEEN WRITTEN TO CSV FILES")



from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import load_iris
import numpy as np 
import pandas as pd 

n_estimators_range = np.arange(20, 100, 20)

max_depth_range = np.arange(5, 30, 5)

param_grid = {
    'n_estimators': n_estimators_range,
    'max_depth': max_depth_range,

}

rf_classifier = RandomForestClassifier(random_state=64)


X = pd.read_csv("/root/katacoda-scenarios/fraud-detection-modeling/assets/features.csv")

y = pd.read_csv("/root/katacoda-scenarios/fraud-detection-modeling/assets/targets.csv")

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=128, test_size = 0.2)

print(X_train.head())
print(y_train.head())


print(X_test.head())
print(y_test.head())

print("DATA HAS BEEN SPLIT FOR TRAINING AND TESTING")
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='precision')

grid_search.fit(X_train, y_train)

print("Best Hyperparameters:", grid_search.best_params_)

best_rf_model = grid_search.best_estimator_

y_pred = best_rf_model.predict(X_test)

print("OPTIMAL MODEL HAS BEEN DEFINED AND PREDICTIONS WERE MADE ON THE TEST SET")

import pickle

validation_data = X_test
validation_data['actual'] = y_test
validation_data['predicted'] = y_pred

validation_data.to_csv("validation_data.csv", index= False)

model_filename = 'random_forest_model.pkl'
with open(model_filename, 'wb') as model_file:
    pickle.dump(best_rf_model, model_file)

print(f"Random Forest model saved to {model_filename}")


import pandas as pd 

validation_data = pd.read_csv("/root/katacoda-scenarios/fraud-detection-evaluate-model/assets/validation_data.csv")

print(validation_data.head())

actual, predicted = validation_data['actual'], validation_data['predicted']

print(actual.head())

print(predicted.head())

print("THE ACTUAL AND PREDICTED VALUES HAVE BEEN READ INTO A DATAFRAME")


from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

precision = precision_score(actual, predicted)

print("precision: ", precision)


accuracy = accuracy_score(actual, predicted)

print("accuracy: ", accuracy)

recall = recall_score(actual, predicted)

print("recall: ", recall)

f1_score = f1_score(actual, predicted)

print("f1_score: ", f1_score)


import pickle

model_filename = '/root/katacoda-scenarios/fraud-detection-evaluate-model/assets/random_forest_model.pkl'
with open(model_filename, 'rb') as model_file:
    loaded_rf_model = pickle.load(model_file)


importances = loaded_rf_model.feature_importances_
features = loaded_rf_model.feature_names_in_

feature_importance_df = pd.DataFrame({"features":features, "importances": importances})
feature_importance_df.sort_values("importances", ascending=False)

print(feature_importance_df)

feature_importance_df.to_csv("feature_importance.csv", index=False)