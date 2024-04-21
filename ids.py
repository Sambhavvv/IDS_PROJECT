
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
data=pd.read_csv("/content/traffic.csv", skiprows=range(1, 25340))
data
null_values = data.isnull().sum()
print(null_values)
features_to_drop = ['wind_speed', 'wind_direction','rain_p_h','snow_p_h','weather_type','date_time']
data.drop(columns=features_to_drop, inplace=True)
data
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()

data['weather_description_encoded'] = ordinal_encoder.fit_transform(data[['weather_description']])
features = ['is_holiday','air_pollution_index','humidity','Unnamed: 15','temperature','visibility_in_miles','dew_point','clouds_all','weather_description_encoded','hour','month_day','month_day']
target = ['traffic_volume']
X = data[features]
y = data[target]
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
X = x_scaler.fit_transform(X)
y = y_scaler.fit_transform(y).flatten()
X
y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
mlp_regressor = MLPRegressor(random_state=1, max_iter=500)
mlp_regressor.fit(X_train, y_train)

y_pred = mlp_regressor.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)
from sklearn.metrics import r2_score
r_squared = r2_score(y_test, y_pred)
print("R-squared (R2):", r_squared)
y_train = np.array(y_train)
gbm_regr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
gbm_regr.fit(X_train, y_train.ravel())
gbm_predictions = gbm_regr.predict(X_test)
gbm_mae = mean_absolute_error(y_test, gbm_predictions)
print("GBM Mean Absolute Error:", gbm_mae)
from sklearn.metrics import r2_score
r_squared = r2_score(y_test, gbm_predictions)
print("R-squared (R2):", r_squared)
gbm_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4)
mlp_model = MLPRegressor(hidden_layer_sizes=(100,))

# Initialize stacking ensemble
estimators = [('gbm', gbm_model), ('mlp', mlp_model)]
stacking_regressor = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())

# Train stacking ensemble
stacking_regressor.fit(X_train, y_train)

# Evaluate performance
stacking_predictions = stacking_regressor.predict(X_test)
stacking_mae = mean_absolute_error(y_test, stacking_predictions)
print("Stacking Ensemble Mean Absolute Error:", stacking_mae)
from sklearn.metrics import r2_score
r_squared = r2_score(y_test, stacking_predictions)
print("R-squared (R2):", r_squared)
