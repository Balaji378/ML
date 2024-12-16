import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load your dataset from a CSV file
data = pd.read_csv('/content/Customers.csv')  # Replace 'your_dataset.csv' with your actual CSV file path

# Inspect the first few rows of the data to understand its structure
print(data.head())

# If 'Gender' is a column, encode it as a numeric value (Male=0, Female=1)
# Modify this line if the column names are different or if you need different encoding
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

# Select the relevant features for clustering (ensure these are the columns you need)
# Modify the column names if necessary (e.g., 'Age', 'Income', 'Spending Score')
features = data[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

# Elbow Method to find the optimal number of clusters (K)
inertia = []
for k in range(1, 61):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.plot(range(1, 61), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.show()

optimal_k = 25  # Adjust this based on the Elbow Method graph
# Based on the Elbow method, choose the optimal number of clusters (say K=3)
kmeans = KMeans(n_clusters=25, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Add the cluster assignments to the data
data['Cluster'] = clusters

# Show the resulting clusters
print(data)

# Save the clustered data to a new CSV file
data.to_csv('clustered_customers.csv', index=False)

# Get the centroids of each cluster
centroids = kmeans.cluster_centers_

# Visualize the clusters (2D plot of the first two features for simplicity)
plt.scatter(data['Spending Score (1-100)'], data['Annual Income (k$)'], c=data['Cluster'], cmap='rainbow', label='Customers')
plt.scatter(centroids[:, 1], centroids[:, 3], c='black', marker='X', s=200, label='Centroids')  # Plot centroids

# Add titles and labels
plt.title(f'Customer Segments Based on Age and Spending Score (K={optimal_k})')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')

# Add legend
plt.legend()

# Show the plot
plt.show()









import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Load your dataset from a CSV file
data = pd.read_csv('/content/Customers.csv')  # Replace with your actual CSV file path

# Inspect the first few rows of the data to understand its structure
print(data.head())

# If 'Gender' is a column, encode it as a numeric value (Male=0, Female=1)
# Modify this line if the column names are different or if you need different encoding
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

# Select the relevant features for clustering (ensure these are the columns you need)
# Modify the column names if necessary (e.g., 'Age', 'Income', 'Spending Score')
features = data[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

# Perform K-Means clustering to get cluster labels
optimal_k = 3 # After visual inspection of the elbow plot, we select the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_data)

# Show the resulting clusters
print(data.head())

# Now, let's use KNN to predict the cluster labels
# We use the scaled data and the 'Cluster' as the target label
X = scaled_data
y = data['Cluster']

# Initialize KNN Classifier
knn = KNeighborsClassifier(n_neighbors=3)  # You can tune the 'n_neighbors' parameter
knn.fit(X, y)  # Train the KNN model using the features and their respective cluster labels

# Predict the clusters for the data (you could use new data points as well)
predicted_clusters = knn.predict(X)

# Add the predicted cluster labels to the data
data['Predicted_Cluster'] = predicted_clusters

# Show the results
print(data.head())

# Save the results with predicted cluster labels to a new CSV file
data.to_csv('clustered_with_knn.csv', index=False)

# Visualize the clusters and the centroids using the KMeans result
centroids = kmeans.cluster_centers_

# 2D plot of Age vs Spending Score with KNN predicted clusters
plt.scatter(data['Age'], data['Spending Score (1-100)'], c=data['Predicted_Cluster'], cmap='rainbow', label='Customers')
plt.scatter(centroids[:, 1], centroids[:, 3], c='black', marker='X', s=200, label='Centroids')  # Plot centroids

# Add titles and labels
plt.title(f'Customer Segments Based on Age and Spending Score (K={optimal_k})')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')

# Add legend
plt.legend()

# Show the plot
plt.show()










import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load your dataset from a CSV file
file_path = '/content/exams.csv'
data = pd.read_csv(file_path)
data.head()

data['gender'] = data['gender'].map({'male': 0, 'female': 1})
#data['test preparation course'] = data['test preparation course'].map({'none': 0, 'completed': 1})
data.head()

features = data[['gender', 'math score', 'reading score', 'writing score']]
features.head()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)
scaled_data

inertia = []
for k in range(1, 21):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)
inertia

plt.plot(range(1,21), inertia, marker='o')
plt.title('elbow result')
plt.xlabel('no.of.clusters')
plt.ylabel('inertia')
plt.show()

optimal_k = 7
kmeans = KMeans(n_clusters= optimal_k, random_state=32)
clusters = kmeans.fit_predict(scaled_data)

data['K']=clusters
data.head()

centroids=kmeans.cluster_centers_

plt.scatter(data['math score'], data['writing score'], c=data['K'], cmap='rainbow', label='Students')
#plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=200, label='Centroids')
plt.title(f'Student Clusters Based on Writing and Reading Scores (K={optimal_k})')
plt.xlabel('Writing Score')
plt.ylabel('Reading Score')
plt.legend()
plt.show()
