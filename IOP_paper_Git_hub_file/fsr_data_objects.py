import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the data from the CSV file
df = pd.read_csv('FSR_data_object_plot.csv')

# Step 2: Set the list of soft objects to plot in red
soft_objects = ['Rubber', 'Thermocol', 'Threadball', 'TPU Box', 'Hollow TPU', 'Green Sponge', 'White Sponge']

# Step 3: Create a single figure for all subplots
plt.figure(figsize=(10, 15))

# Step 4: Plot the FSR data for each object
for idx, column in enumerate(df.columns, start=1):
    plt.subplot(5, 3, idx)  # Adjust subplot indices for 3 rows and 5 columns

    # Determine color based on the object type
    color = 'red' if column in soft_objects else 'blue'

    # Plot FSR data
    plt.plot(df[column], color=color)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('F-volt', fontsize=12)
    plt.title(column, fontsize=14)
    plt.grid()

# Adjust layout to avoid overlapping
plt.tight_layout(pad=2.5, h_pad=1.5, w_pad=0.5)

# Save the plot
plt.savefig('soft_objects_plot.png', dpi=300)

plt.show()
