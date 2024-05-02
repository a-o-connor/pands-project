# Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data set

df = pd.read_csv("https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv")

# Print the summary statistics of all the variables of the data set

print(df.describe())
print(df.info())

#Create a dataframe with only the numeric variables:
df_numeric = df.select_dtypes(include = ('number'))

#View the correlations between the continuous, numeric variables 
print(df_numeric.corr())

#Save a histogram of each numeric variable as a .png

fig, ax = plt.subplots()
ax.hist(df_numeric["sepal_length"], edgecolor = "black")
ax.set_xlabel("Sepal Length (mm)")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Sepal Length in Iris Data Set")
plt.savefig("Histogram Sepal Length")


fig1, ax1 = plt.subplots()
ax1.hist(df_numeric["sepal_width"], edgecolor = "black")
ax1.set_xlabel("Sepal Width (mm)")
ax1.set_ylabel("Frequency")
ax1.set_title("Distribution of Sepal Width in Iris Data Set")
plt.savefig("Histogram Sepal Width")

fig2, ax2 = plt.subplots()
ax2.hist(df_numeric["petal_length"], edgecolor = "black")
ax2.set_xlabel("Petal Length (mm)")
ax2.set_ylabel("Frequency")
ax2.set_title("Distribution of Petal Length in Iris Data Set")
plt.savefig("Histogram Petal Length")

fig3, ax3 = plt.subplots()
ax3.hist(df_numeric["petal_width"], edgecolor = "black")
ax3.set_xlabel("Petal Width (mm)")
ax3.set_ylabel("Frequency")
ax3.set_title("Distribution of Petal Width in Iris Data Set")
plt.savefig("Histogram Petal Width")

# Plot a bar chart of mean Sepal  length, Sepal Width, Petal Width and Petal length between the species using Matplotlib
# Plot the three variables on the same bar chart
# Plot a bar chart of mean Sepal  length, Sepal Width, Petal Width and Petal length between the species using Matplotlib
# Plot the three variables on the same bar chart

species = list(df["species"].unique()) # returns a np array with the three species in the data frame
species_means = {
    "Sepal Length": list(df.groupby(["species"])["sepal_length"].mean()), # Returns a Pandas series with the mean of sepal length by species
    "Sepal Width": list(df.groupby(["species"])["sepal_width"].mean()),
    "Petal Length": list(df.groupby(["species"])["petal_length"].mean()),
    "Petal Width": list(df.groupby(["species"])["petal_width"].mean())
}

x = np.arange(len(species))  # returns a numpy array same length as the species list created earlier

fig, ax = plt.subplots()

ax.bar(x+0.2, species_means["Sepal Length"], width = 0.2, label = "Sepal Length") # x values offset by 0.2 (width of the bars)
ax.bar(x+0.4, species_means["Sepal Width"], width = 0.2, label = "Sepal Width")
ax.bar(x, species_means["Petal Length"], width = 0.2, label="Petal Length")
ax.bar(x+0.6, species_means["Petal Width"], width = 0.2, label="Petal Width")

ax.set_ylabel('Length/ Width (mm)') #Label y axis 
ax.set_xlabel('Species') #Label x axis
ax.set_title('Iris Sepal and Petal Length and Width by Species') # Set title of bar chart 
ax.set_xticks(x + 0.3, species) # Labels and ticks on the x axis, offset so that the ticks sit in the middle of the 4 bars
fig.legend(loc = "upper right", bbox_to_anchor=(-0.13, 0.38, 0.5, 0.5)) # Setting the location of the legend outside of the plot.
plt.savefig("barchart")

# Save a scatteerplot of of each pair of variables
fig5, ax5 = plt.subplots()
ax5.plot(df_numeric["sepal_length"], df_numeric["petal_length"], "o")
ax5.set_xlabel("Sepal Length (mm)")
ax5.set_ylabel("Petal Length (mm)")
ax5.set_title("Sepal Length vs. Petal Length in Iris Flowers")
plt.savefig("Scatterplot Sepal Length vs Petal Length")



