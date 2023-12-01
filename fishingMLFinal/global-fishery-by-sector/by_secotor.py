import matplotlib.pyplot as plt
import pandas as pd

by_sector_path = 'global-fishery-catch-by-sector.csv'
fishery_data = pd.read_csv(by_sector_path)
# fishery_data.head()

plt.figure(figsize=(12, 8))

# info
plt.plot(fishery_data['Year'], fishery_data['Artisanal (small-scale commercial)'], label='Artisanal')
plt.plot(fishery_data['Year'], fishery_data['Discards'], label='Discards')
plt.plot(fishery_data['Year'], fishery_data['Industrial (large-scale commercial)'], label='Industrial')
plt.plot(fishery_data['Year'], fishery_data['Recreational'], label='Recreational')
plt.plot(fishery_data['Year'], fishery_data['Subsistence'], label='Subsistence')

# labels
plt.xlabel('Year')
plt.ylabel('Catch (in metric tons)')
plt.title('Global Fishery Catch by Sector Over Time')
plt.legend()

plt.grid(True)
plt.show()
