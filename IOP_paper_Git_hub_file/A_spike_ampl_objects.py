import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.cluster import KMeans

# Step 1: Load the data from the CSV file
# Replace 'selected_features.csv' with the actual filename/path
df = pd.read_csv('selected_features.csv')

# Step 2: Generate time values as a sequence
time_interval = 0.001  # Replace with the actual time interval between data points
time = np.arange(0, len(df) * time_interval, time_interval)

# Step 3: Set Seaborn style and color palette
sns.set(style='whitegrid')
colors = sns.color_palette()

# Step 4: Create a single figure for all subplots
plt.figure(figsize=(10, 15))

# Step 5: Apply KMeans clustering
#n_clusters = 2  # Replace with the desired number of clusters
#kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#clusters = kmeans.fit_predict(df)

# Columns to plot in red
columns_to_plot_red = ['Rubber', 'Thermocol', 'Threadball', 'TPU', 'Hollow TPU', 'Green Sponge', 'White Sponge']

# Step 6: Calculate and plot the raw data for each column
for idx, column in enumerate(df.columns, start=1):
    plt.subplot(5, 3, idx)  # Adjust subplot indices for 5 rows and 3 columns

    # Determine color based on the column name
    color = 'red' if column in columns_to_plot_red else 'blue'

    # Plot only positive values
    positive_values = df[df[column] > 0][column]
    time_positive = time[:len(positive_values)]  # Adjust time array for positive values

    # Plot the raw data
    plt.plot(time_positive, positive_values, color=color)
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Amplitude', fontsize=16)
    plt.title(f'{column}', fontsize=18)
    plt.grid()

# Adjust layout to avoid overlapping
plt.tight_layout(pad=2.5, h_pad=1.5, w_pad=0.5)

# Rotate x-axis labels for the second row of subplots
plt.xticks(rotation=45)

# Save the raw data plot in HD
# Replace 'raw_data_plot_hd_positive.png' with the desired filename/path for the HD image
plt.savefig('raw_data_plot_hd_positive.png', dpi=300)  # Set dpi to 300 for HD image

plt.show()
