import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler
# Load data
file_path = 'exams.csv'  # Path to the dataset
df = pd.read_csv(file_path)
# Select relevant columns
scores = df[['math score', 'reading score', 'writing score']]
# Standardize the data
scaler = StandardScaler()
scaled_scores = scaler.fit_transform(scores)
# Perform hierarchical clustering
linked = linkage(scaled_scores, method='ward')
# Plot dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked, labels=df.index)
plt.title('Dendrogram of Exam Scores')
plt.xlabel('Students')
plt.ylabel('Euclidean distances')
plt.show()

num_clusters = 4
clusters = fcluster(linked, num_clusters, criterion='maxclust')

# Add clusters to the dataframe
df['Cluster'] = clusters

# Visualize clusters using pairplot
sns.pairplot(df, vars=['math score', 'reading score', 'writing score'], hue='Cluster', palette='viridis')
plt.show()
