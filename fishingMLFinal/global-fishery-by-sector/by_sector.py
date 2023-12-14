import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

font_size = 12

by_sector_path = 'global-fishery-catch-by-sector.csv'
fishery_data = pd.read_csv(by_sector_path)

# Sectors to analyze
sectors = ['Artisanal (small-scale commercial)', 'Discards',
           'Industrial (large-scale commercial)', 'Recreational', 'Subsistence']

# Loop through each sector and apply models
for sector in sectors:
    # Set up
    plt.figure(num=f'Global Fishery Catch by {sector}', figsize=(12, 8))
    plt.xlabel('Year')
    plt.ylabel('Catch (in metric tons)')
    plt.title('ML Models for Global Fishery Catch by ' + sector)

    # Prepare the data
    X = fishery_data['Year'].values.reshape(-1, 1)
    y = fishery_data[sector]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Linear Regression model
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)

    # k-NN model
    model_knn = KNeighborsRegressor(n_neighbors=3)
    model_knn.fit(X_train, y_train)

    # Decision Tree model
    model_dt = DecisionTreeRegressor()
    model_dt.fit(X_train, y_train)

    # Predictions
    y_pred_lr = model_lr.predict(X_test)
    y_pred_knn = model_knn.predict(X_test)
    y_pred_dt = model_dt.predict(X_test)

    # Sort
    sorted_indices = np.argsort(X_test.ravel())
    X_test_sorted = X_test[sorted_indices]
    y_pred_lr_sorted = y_pred_lr[sorted_indices]
    y_pred_knn_sorted = y_pred_knn[sorted_indices]
    y_pred_dt_sorted = y_pred_dt[sorted_indices]

    # Plot
    plt.scatter(X, y, label=f'{sector} Data')
    lr_line, = plt.plot(X_test_sorted, y_pred_lr_sorted, label=f'{sector} Linear Regression', linestyle='solid')
    knn_line, = plt.plot(X_test_sorted, y_pred_knn_sorted, label=f'{sector} k-NN Regression', linestyle='dotted')
    dt_line, = plt.plot(X_test_sorted, y_pred_dt_sorted, label=f'{sector} Decision Tree Regression', linestyle='dashdot')

    # MSE for each model
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    mse_knn = mean_squared_error(y_test, y_pred_knn)
    mse_dt = mean_squared_error(y_test, y_pred_dt)

    # Display MSE for each model on the graph (bottom left)
    plt.text(0.02, 0.05, f'LR MSE: {int(mse_lr):,}', transform=plt.gca().transAxes, fontsize=font_size,color=lr_line.get_color())
    plt.text(0.02, 0.10, f'k-NN MSE: {int(mse_knn):,}', transform=plt.gca().transAxes, fontsize=font_size,color=knn_line.get_color())
    plt.text(0.02, 0.15, f'DT MSE: {int(mse_dt):,}', transform=plt.gca().transAxes, fontsize=font_size,color=dt_line.get_color())

    # Add text for the final prediction value of each model in the bottom right
    plt.text(0.98, 0.05, f'LR Prediction: {int(y_pred_lr_sorted[-1]):,}', horizontalalignment='right', transform=plt.gca().transAxes,
             fontsize=font_size,color=lr_line.get_color())
    plt.text(0.98, 0.10, f'k-NN Prediction: {int(y_pred_knn_sorted[-1]):,}', horizontalalignment='right', transform=plt.gca().transAxes,
             fontsize=font_size,color=knn_line.get_color())
    plt.text(0.98, 0.15, f'DT Prediction: {int(y_pred_dt_sorted[-1]):,}', horizontalalignment='right', transform=plt.gca().transAxes,
             fontsize=font_size,color=dt_line.get_color())

    plt.legend()
    plt.grid(True)

plt.show()