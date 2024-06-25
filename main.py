import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# %%

data = pd.read_csv("D:/Datasets/House Price/House_Price_dataset.csv")
data.drop(data.columns[[0, 1, 2, 5, 6, 7, 11, 14, 15, 16, 19]], inplace=True, axis=1)

X = data.drop('price', axis=1)
y = np.array(data.iloc[::, 1]).reshape((-1,))

X['purpose'] = LabelEncoder().fit_transform(np.array(data['purpose']).reshape((-1,)))
X['Area Type'] = LabelEncoder().fit_transform(np.array(data['Area Type']).reshape((-1,)))

columnTransformer = ColumnTransformer(transformers=[
    ('min-max', MinMaxScaler(), ['latitude', 'longitude']),
    ('encoder', OneHotEncoder(sparse_output=False), ['property_type', 'baths', 'bedrooms']),
], remainder='passthrough')

X = columnTransformer.fit_transform(X)

X = X.astype(np.float64)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

lr = RandomForestRegressor(n_jobs=-1, verbose=2, n_estimators=200)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print("r2 score:", r2_score(y_test, y_pred))

# %%

plt.figure(figsize=(18, 19.8))
plt.title('Random Forest Regression Sample Result Analysis')
plt.plot(y_test[:100], label='Actual')
plt.plot(y_pred[:100], marker='o', label='Predicted')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()
