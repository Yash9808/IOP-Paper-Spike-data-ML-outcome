import matplotlib.pyplot as plt

# Accuracies data
fsr_accuracies = {
    'Logistic Regression': [98, 98, 98, 98, 98],
    'Random Forest': [98, 97, 97, 98, 98],
    'Decision Tree': [98, 97, 98, 98, 98],
    'SVC': [98, 98, 98, 98, 98],
    'MLP': [98, 96, 97, 96, 98],
    'KNN':[98,98,98,98,98]
}

neuron_accuracies = {
    'Random Forest': [99.9, 99.9, 99.9, 99.9, 99.9],
    'Decision Tree': [100.0, 100.0, 100.0, 100.0, 100.0],
    'Logistic Regression': [80.9, 80.7, 80.72, 80.59, 80.58],
    'MLP': [100.0, 99.9, 99.98, 99.94, 92.81],
    'SVC': [96.26, 96.08, 96.15, 96.23, 96.22],
    'KNN':[100,100,100,100,100]
}

test_sizes = [20, 40, 50, 70, 90]

# Plotting
fig, axs = plt.subplots(3, 2, figsize=(15, 15))
fig.subplots_adjust(hspace=0.5)

fsr_classifiers = list(fsr_accuracies.keys())

for i, ax in enumerate(axs.flat):
    fsr_classifier_name = fsr_classifiers[i]
    fsr_classifier_accuracies = fsr_accuracies[fsr_classifier_name]
    neuron_classifier_accuracies = neuron_accuracies[fsr_classifier_name]

    ax.plot(test_sizes, fsr_classifier_accuracies, label='FSR', marker='o', linestyle='-', color='black')
    ax.plot(test_sizes, neuron_classifier_accuracies, label='Neuron:Pt/Ag:SiOx/Pt', marker='o', linestyle='-', color='red')

    ax.set_title(f'{fsr_classifier_name} Accuracies', fontsize=14)
    ax.set_xlabel('Test Size (%)', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_xticks(test_sizes)
    ax.legend()
    ax.grid(True)

plt.show()
