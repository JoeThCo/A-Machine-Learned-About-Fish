import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import patheffects
from sklearn.linear_model import LinearRegression
import numpy as np
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

    # Linear regression (limited to 2000-2010)
    lr_filter = (fishery_data['Year'] >= 2000) & (fishery_data['Year'] <= 2010)
    X_lr = fishery_data[lr_filter]['Year'].values.reshape(-1, 1)
    y_lr = fishery_data[lr_filter][sector]

    model_lr = LinearRegression()
    model_lr.fit(X_lr, y_lr)

    # k-NN regression
    model_knn = KNeighborsRegressor(n_neighbors=3)
    model_knn.fit(X, y)

    # Predictions for Linear Regression (extrapolation)
    X_extended_lr = np.arange(2010, 2015).reshape(-1, 1)
    y_prediction_lr_extended = model_lr.predict(X_extended_lr)

    # Predictions for k-NN (entire range)
    y_prediction_knn = model_knn.predict(X)

    # plot points
    plt.scatter(X, y)

    # Plot the Linear Regression extrapolation (2010-2015)
    line, = plt.plot(X_extended_lr, y_prediction_lr_extended, linestyle='dashed')
    line_color = line.get_color()

    # Plot the k-NN regression line
    plt.plot(X, y_prediction_knn, label=f'{sector} k-NN Regression', linestyle='solid', color=line_color)

    # labels
    predicted_value_2015_lr = y_prediction_lr_extended[-1]
    formatted_value_2015_lr = '{:,}'.format(int(predicted_value_2015_lr))
    text = plt.text(2015, predicted_value_2015_lr, formatted_value_2015_lr, fontsize=16, verticalalignment='bottom',color=line_color)
    text.set_path_effects([patheffects.withStroke(linewidth=1, foreground='black')])

# labels
plt.xlabel('Year')
plt.ylabel('Catch (in metric tons)')
plt.title('Predicting Global Fishery Catch by Sector')
plt.legend()

plt.grid(True)
plt.show()
