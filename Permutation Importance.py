# Machine-Learning-Explainability
#setup
# Loading data, dividing, modeling and EDA below
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
data = pd.read_csv('../input/new-york-city-taxi-fare-prediction/train.csv', nrows=50000)
# Remove data with extreme outlier coordinates or negative fares
data = data.query('pickup_latitude > 40.7 and pickup_latitude < 40.8 and ' +
                  'dropoff_latitude > 40.7 and dropoff_latitude < 40.8 and ' +
                  'pickup_longitude > -74 and pickup_longitude < -73.9 and ' +
                  'dropoff_longitude > -74 and dropoff_longitude < -73.9 and ' +
                  'fare_amount > 0'
                  )

y = data.fare_amount
base_features = ['pickup_longitude',
                 'pickup_latitude',
                 'dropoff_longitude',
                 'dropoff_latitude',
                 'passenger_count']
                 
X = data[base_features]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
first_model = RandomForestRegressor(n_estimators=50, random_state=1).fit(train_X, train_y)

# Environment Set-Up for feedback system.
from learntools.core import binder
binder.bind(globals())
from learntools.ml_explainability.ex2 import *
print("Setup Complete")
# show data
print("Data sample:")
data.head()

train_X.describe()
train_y.describe()
#which variables seem potentially useful for predicting taxi fares?: The features that seem important are :pickup_longitude;pickup_latitude;dropoff_longitude;dropoff_latitude But NY taxis might take into consideration the number of passengers unlike what's normally done so a permutation importance seems necessary
import eli5
from eli5.sklearn import PermutationImportance
perm=PermutationImportance(first_model,random_state=1).fit(val_X,val_y)
eli5.sow_weights(perm,features_names=val_X.columns.tolist())
#Latitude matters more because generally in one taxi travel  the latitude is what varies much more than longitude values

# create new features
data['abs_lon_change'] = abs(data.dropoff_longitude - data.pickup_longitude)
data['abs_lat_change'] = abs(data.dropoff_latitude - data.pickup_latitude)

features_2  = ['pickup_longitude',
               'pickup_latitude',
               'dropoff_longitude',
               'dropoff_latitude',
               'abs_lat_change',
               'abs_lon_change']
X = data[features_2]
new_train_X, new_val_X, new_train_y, new_val_y = train_test_split(X, y, random_state=1)
second_model = RandomForestRegressor(n_estimators=30, random_state=1).fit(new_train_X, new_train_y)
# Create a PermutationImportance object on second_model and fit it to new_val_X and new_val_y
# Use a random_state of 1 for reproducible results that match the expected solution.
perm2 = PermutationImportance(second_model,random_state=1).fit(new_val_X,new_val_y)
# show the weights for the permutation importance you just calculated
eli5.show_weights(perm2,feature_names=new_val_X.columns.tolist())

#abs_lon_change and abs_lat_change are pretty small had smaller importance. However the scale of features does not affect permutation importance.
#The importance for latitudinal distance is greater than the importance of longitudinal distance,however We cannot tell f whether traveling a fixed latitudinal distance is more or less expensive than traveling the same longitudinal distance.
