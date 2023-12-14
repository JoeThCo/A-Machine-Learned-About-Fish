import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from matplotlib import patheffects
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

by_sector_path = 'global-fishery-catch-by-sector.csv'
sectors = ['Artisanal (small-scale commercial)', 'Discards', 'Industrial (large-scale commercial)', 'Recreational',
           'Subsistence']

fishery_data = pd.read_csv(by_sector_path)
plt.figure(figsize=(12, 8))

# Loop through each sector and perform linear regression
for sector in sectors:
    # prep the data
    X = fishery_data['Year'].values.reshape(-1, 1)
    y = fishery_data[sector]

    # standard data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Linear regression (limited to 2000-2010)
    lr_filter = (fishery_data['Year'] >= 2000) & (fishery_data['Year'] <= 2010)
    X_lr = fishery_data[lr_filter]['Year'].values.reshape(-1, 1)
    y_lr = fishery_data[lr_filter][sector]

    model_lr = LinearRegression()
    model_lr.fit(X_lr, y_lr)

    # k-NN regression
    model_knn = KNeighborsRegressor(n_neighbors=3)
    model_knn.fit(X_scaled, y)

    # Predictions for Linear Regression
    X_extended_lr = np.arange(2010, 2015).reshape(-1, 1)
    y_prediction_lr_extended = model_lr.predict(X_extended_lr)

    # Predictions for k-NN
    y_prediction_knn = model_knn.predict(X_scaled)

    # Plot points
    plt.scatter(X, y)

    # Plot Lines
    line, = plt.plot(X_extended_lr, y_prediction_lr_extended, linestyle='dashed')
    line_color = line.get_color()

    plt.plot(X, y_prediction_knn, label=f'{sector} k-NN Regression', linestyle='solid', color=line_color)
    predicted_value_2015_lr = y_prediction_lr_extended[-1]
    formatted_value_2015_lr = '{:,}'.format(int(predicted_value_2015_lr))
    text = plt.text(2015, predicted_value_2015_lr, formatted_value_2015_lr, fontsize=16, verticalalignment='bottom',color=line_color)
    text.set_path_effects([patheffects.withStroke(linewidth=1, foreground='black')])

# Chart labels
plt.xlabel('Year')
plt.ylabel('Catch (in metric tons)')
plt.title('Predicting Global Fishery Catch by Sector')
plt.legend()

plt.grid(True)
plt.show()

# Intro
# ML techniques

# Term Project Rules
# Make a team
# Pick ML algorithums
# Show results thought cross validation (How to show this?)
# Report

# Report
# Description of ML algorithim I picked for the dataset
# Data visual
# data cleaning
# 3 ML techniques or models you
# Picked the best one
# Validated the results and metrics for model assignment