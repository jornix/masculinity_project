import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

survey = pd.read_csv("masculinity.csv")
pd.set_option('display.max_columns', None)
'''
# The Masculinity survey.
First, let's take a quick look at the first five rows of raw data:
'''
st.write(survey.head())

cols_to_map = ["q0007_0001", "q0007_0002", "q0007_0003", "q0007_0004",
       "q0007_0005", "q0007_0006", "q0007_0007", "q0007_0008", "q0007_0009",
       "q0007_0010", "q0007_0011"]

for col in cols_to_map:
    survey[col]=survey[col].map({"Often":4, "Sometimes":3, "Rarely":2, "Never, but open to it":1,"Never, and not open to it":0})




'''
## Plotting the Plot
We'll be working on the sub questions for question 7.
- Ask a friend for professional advice (q0007_0001)
- Ask a friend for personal advice (q0007_0002)
- Express physical affection to male friends, like hugging, rubbing shoulders (q0007_0003)
- Cry (q0007_0004)
- Get in a physical fight with another person (q0007_0005)
- Have sexual relations with women, including anything from kissing to sex (q0007_0006)
- Have sexual relations with men, including anything from kissing to sex (q0007_0007)
- Watch sports of any kind (q0007_0008)
- Work out (q0007_0009)
- See a therapist (q0007_0010)
- Feel lonely or isolated (q0007_0011)
The participants rated themselves from *Often* (4), *Sometimes* (3), *Rarely* (2), *Never, but open to it* (1) or *Never, and not open to it* (0).

Select columns to plot:
'''
x = st.selectbox('Select X', cols_to_map)
y = st.selectbox('Select y', cols_to_map)

fig,ax = plt.subplots()
ax.scatter(survey[x],survey[y],c='r',alpha=0.1)
plt.title("Plot of columns "+x+" and "+y+".")
plt.xlabel(x)
plt.ylabel(y)
st.pyplot(fig)

'''
## Build a KMeans model
'''
rows_to_cluster = survey.dropna(subset=["q0007_0001", "q0007_0002", "q0007_0003", "q0007_0004","q0007_0005","q0007_0008","q0007_0009"])

classifier = KMeans(n_clusters=2)
classifier.fit(rows_to_cluster[["q0007_0001", "q0007_0002", "q0007_0003", "q0007_0004","q0007_0005","q0007_0008","q0007_0009"]])
'''
Cluster centers:
'''
st.write(classifier.cluster_centers_)

cluster_zero_indices = []
cluster_one_indices = []

for i in range(len(classifier.labels_)):
    if classifier.labels_[i] == 0:
        cluster_zero_indices.append(i)
    elif classifier.labels_[i] == 1:
        cluster_one_indices.append(i)

cluster_zero_df = rows_to_cluster.iloc[cluster_zero_indices]
cluster_one_df = rows_to_cluster.iloc[cluster_one_indices]

'''
Notice how the respondees differ by education in these clusters:
'''
st.write(cluster_zero_df["educ4"].value_counts()/len(cluster_zero_df))
st.write(cluster_one_df["educ4"].value_counts()/len(cluster_one_df))

# st.write(cluster_zero_df["kids"].value_counts()/len(cluster_zero_df))
# st.write(cluster_one_df["kids"].value_counts()/len(cluster_one_df))
#
# st.write(cluster_zero_df["racethn4"].value_counts()/len(cluster_zero_df))
# st.write(cluster_one_df["racethn4"].value_counts()/len(cluster_one_df))
#
# st.write(cluster_zero_df["q0036"].value_counts()/len(cluster_zero_df))
# st.write(cluster_one_df["q0036"].value_counts()/len(cluster_one_df))
'''
... but not in how they report their feeling of masculinity:
'''
st.write(cluster_zero_df["q0001"].value_counts()/len(cluster_zero_df))
st.write(cluster_one_df["q0001"].value_counts()/len(cluster_one_df))
