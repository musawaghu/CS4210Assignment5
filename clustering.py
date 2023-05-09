# -------------------------------------------------------------------------
# AUTHOR: Musa Waghu
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #5
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/

# importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

df = pd.read_csv('training_data.csv', sep=',', header=None)  # reading the data by using Pandas library

# assign your training data to X_training feature matrix
X_training = df.values
# run kmeans testing different k values from 2 until 20 clusters
# Use:  kmeans = KMeans(n_clusters=k, random_state=0)
#      kmeans.fit(X_training)
silhouette_scores = []
k_values = range(2, 21)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_training)
    silhouette_scores.append(silhouette_score(X_training, kmeans.labels_))



# for each k, calculate the silhouette_coefficient by using: silhouette_score(X_training, kmeans.labels_)
# find which k maximizes the silhouette_coefficient
max_k = k_values[np.argmax(silhouette_scores)]

# plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
plt.plot(k_values, silhouette_scores)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Coefficient')
plt.title('Silhouette Coefficient for each k value')
plt.show()

# reading the test data (clusters) by using Pandas library
kmeans = KMeans(n_clusters=max_k, random_state=0)
kmeans.fit(X_training)

# assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
# do this: np.array(df.values).reshape(1,<number of samples>)[0]
TestDF = pd.read_csv('testing_data.csv', sep=',', header=None)
data_labels = np.array(TestDF.values).reshape(1, -1)[0]
labels_pred = kmeans.labels_

# Calculate and print the Homogeneity of this kmeans clustering
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(data_labels, labels_pred).__str__())
# --> add your Python code here
