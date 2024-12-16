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



import math
import csv

# Helper functions for ID3 algorithm

# 1. Calculate Entropy
def calculate_entropy(data):
    total = len(data)
    label_counts = {}
    
    # Count the frequency of each label
    for record in data:
        label = record[-1]  # The label is the last column
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1
    
    # Calculate entropy
    entropy = 0
    for count in label_counts.values():
        prob = count / total
        entropy -= prob * math.log2(prob)
    
    return entropy

# 2. Calculate Information Gain for a particular feature
def calculate_info_gain(data, feature_index):
    total_entropy = calculate_entropy(data)
    
    # Get unique values for the chosen feature
    feature_values = set(record[feature_index] for record in data)
    
    weighted_entropy = 0
    for value in feature_values:
        subset = [record for record in data if record[feature_index] == value]
        subset_entropy = calculate_entropy(subset)
        weighted_entropy += (len(subset) / len(data)) * subset_entropy
    
    # Information Gain = Total Entropy - Weighted Entropy of the feature
    return total_entropy - weighted_entropy

# 3. Choose the best feature to split the data
def choose_best_feature(data):
    num_features = len(data[0]) - 1  # Exclude the label column
    best_gain = -1
    best_feature = -1
    
    # For each feature, calculate the information gain
    for feature_index in range(num_features):
        info_gain = calculate_info_gain(data, feature_index)
        if info_gain > best_gain:
            best_gain = info_gain
            best_feature = feature_index
    
    return best_feature

# 4. Build the Decision Tree recursively
def build_tree(data, features):
    # If all the records have the same label, return that label as the leaf node
    labels = [record[-1] for record in data]
    if len(set(labels)) == 1:
        return labels[0]
    
    # If no features left to split on, return the majority label
    if not features:
        return max(set(labels), key=labels.count)
    
    # Choose the best feature to split on
    best_feature = choose_best_feature(data)
    best_feature_name = features[best_feature]
    
    # Create a tree node with the best feature
    tree = {best_feature_name: {}}
    
    # Split the dataset based on the best feature
    feature_values = set(record[best_feature] for record in data)
    
    for value in feature_values:
        # Create a subset of data where the best feature equals the current value
        subset = [record for record in data if record[best_feature] == value]
        
        # Recursively build the tree for the subset
        subtree = build_tree(subset, [f for f in features if f != best_feature])
        
        # Add the subtree to the tree
        tree[best_feature_name][value] = subtree
    
    return tree

# 5. Classify a new instance based on the decision tree
def classify(tree, instance):
    if isinstance(tree, str):
        return tree  # If it's a leaf node, return the label
    
    # Get the feature and value to split on
    feature = list(tree.keys())[0]
    feature_value = instance[0]  # The instance is an array with values for each feature
    
    # Follow the branch corresponding to the feature's value
    return classify(tree[feature][feature_value], instance[1:])

# Load dataset (for simplicity, assume the dataset is in a CSV format)
def load_data_from_csv(filename):
    data = []
    features = []
    
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Read header
        features = header[:-1]  # All columns except the last one (label column)
        
        for row in reader:
            data.append([float(x) if x.replace('.', '', 1).isdigit() else x for x in row])
    
    return data, features

# Example usage

# Load dataset
filename = 'fruits.csv'  # Example CSV file name
data, features = load_data_from_csv(filename)

# Build the decision tree using ID3
tree = build_tree(data, features)

# Print the tree
print("Decision Tree:")
print(tree)

# Predict the class of a new instance
new_instance = [65, 4.0]  # Example new fruit: weight = 180g, size = 8cm
prediction = classify(tree, new_instance)
print(f"Prediction for new instance {new_instance}: {prediction}")
