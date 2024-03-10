# %%
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing

# %% [markdown]
# # Univariate Outlier detection

# %%
df = pd.read_csv('prog_book.csv')

# %%
df.info(), df.describe(), df.head()

# %%
df.boxplot()

# %%
pages_outlier_indices = np.where(df["Number_Of_Pages"] > df["Number_Of_Pages"].quantile(0.75) + 1.5 * (df["Number_Of_Pages"].quantile(0.75) - df["Number_Of_Pages"].quantile(0.25)))[0]

price_outlier_indices = np.where(df["Price"] > df["Price"].quantile(0.75) + 1.5 * (df["Price"].quantile(0.75) - df["Price"].quantile(0.25)))[0]

# %%
print(df.iloc[price_outlier_indices]["Price"]) # price outliers

# %%
print(df.iloc[pages_outlier_indices]["Number_Of_Pages"]) # pages outliers

# %% [markdown]
# # Multivariate Outlier detection

# %%
columns_of_interest = ["Price", "Number_Of_Pages", "Rating", "Reviews", "Type"]

# %%
if type(df["Reviews"].values[0]) == str:
    df["Reviews"] = df["Reviews"].apply(lambda x : int(x.replace(",", "")))
if type(df["Type"].values[0] != int):
    df["Type"] = df["Type"].replace({'Hardcover' : 1, 'Kindle Edition' : 2, 'Paperback' : 3, 'ebook' : 4,
       'Unknown Binding' : 5, 'Boxed Set - Hardcover' : 6})
    
df[columns_of_interest] = pd.DataFrame(preprocessing.StandardScaler().fit(np.array(df[columns_of_interest])).transform(np.array(df[columns_of_interest])))

# %%
def plot_DBSCAN(dbscan_fit, original_data):
    labels = dbscan_fit.labels_
    # print(labels)
    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[dbscan_fit.core_sample_indices_] = True

    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
        class_member_mask = labels == k
        xy = original_data[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

        xy = original_data[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    plt.title(f"Estimated number of clusters: {n_clusters_}")
    plt.show()

# %%
for coli in columns_of_interest:
    for colj in columns_of_interest:
        if coli == colj:
            continue
        dbscan_fit = DBSCAN(eps=0.5, min_samples=2).fit(df[[coli, colj]])
        print(f"~~~~~~~~~~~~~~ PLOTTING FOR COLUMNS '{coli}' AND '{colj}' ~~~~~~~~~~~~~~~~~~~~~")
        plot_DBSCAN(dbscan_fit, np.array(df[[coli, colj]]))
        print("~~~~~~~~~~~~~~ CORRESPONDING OUTLIERS ~~~~~~~~~~~~~~")
        print("NUMBER OF OUTLIERS : ", len(np.where(dbscan_fit.labels_ == -1)[0]))
        print(df[["Book_title", coli, colj]].iloc[np.where(dbscan_fit.labels_ == -1)[0]])
        

# %%
for coli in columns_of_interest:
    for colj in columns_of_interest:
        for colk in columns_of_interest:
            if coli == colj or colj == colk or colk == coli:
                continue

            # Apply DBSCAN
            dbscan = DBSCAN(eps=0.9, min_samples=10)
            clusters = dbscan.fit_predict(np.array(df[[coli, colj, colk]]))

            # Plotting
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Get unique labels
            unique_labels = set(clusters)

            # Colors for the clusters
            colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

            # Plot the points with colors
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    # Black used for noise.
                    col = 'k'

                class_member_mask = (clusters == k)

                xyz = np.array(df[[coli, colj, colk]])[class_member_mask]
                ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=[col], s=50)

            ax.set_title('DBSCAN clustering')
            ax.set_xlabel(coli)
            ax.set_ylabel(colj)
            ax.set_zlabel(colk)
            print(f"~~~~~~~~~~~~~~ PLOTTING FOR COLUMNS '{coli}', '{colj}' AND '{colk}' ~~~~~~~~~~~~~~~~~~~~~")
            plt.show()
            print("~~~~~~~~~~~~~~ CORRESPONDING OUTLIERS ~~~~~~~~~~~~~~")
            print("NUMBER OF OUTLIERS : ", len(np.where(clusters == -1)[0]))
            print(df[["Book_title", coli, colj, colk]].iloc[np.where(clusters == -1)[0]])



