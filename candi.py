# Candidate Elimination Algorithm Implementation

import pandas as pd

# Load the dataset
file_path = '/content/apple.csv'
weather_data = pd.read_csv(file_path)

# Extract features and target
X = weather_data.iloc[:, :-1].values  # All columns except the last
y = weather_data.iloc[:, -1].values   # Last column as target

# Initialize specific and general boundaries
specific_hypothesis = ['0'] * X.shape[1]
general_hypothesis = [['?'] * X.shape[1]]

# Function to update the hypotheses
def candidate_elimination(X, y):
    global specific_hypothesis, general_hypothesis
    for i, instance in enumerate(X):
        if y[i] == 'yes':
            # Update specific hypothesis
            for j in range(len(specific_hypothesis)):
                if specific_hypothesis[j] == '0':
                    specific_hypothesis[j] = instance[j]
                elif specific_hypothesis[j] != instance[j]:
                    specific_hypothesis[j] = '?'
            # Update general hypothesis
            general_hypothesis = [g for g in general_hypothesis if all(
                g[k] == '?' or g[k] == instance[k] for k in range(len(instance)))]
        else:
            # Update general hypothesis
            new_general = []
            for g in general_hypothesis:
                for j in range(len(instance)):
                    if g[j] == '?':
                        temp = g.copy()
                        temp[j] = instance[j] if specific_hypothesis[j] != '?' else '?'
                        new_general.append(temp)
            general_hypothesis.extend(new_general)
            general_hypothesis = [g for g in general_hypothesis if g != specific_hypothesis]

# Run the algorithm
candidate_elimination(X, y)

# Print final hypotheses
print("Specific Hypothesis:", specific_hypothesis)
print("General Hypothesis:", general_hypothesis)
