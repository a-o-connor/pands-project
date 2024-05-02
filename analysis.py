##################################
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

###################################
# Load the data set
df = pd.read_csv("https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv")

###################################
# Print the summary statistics of all the variables of the data set
print(df.describe())
print(df.info())

###################################
#Create a dataframe with only the numeric variables:
df_numeric = df.select_dtypes(include = ('number'))
#View the correlations between the continuous, numeric variables 
print(df_numeric.corr())

###################################
#Save a histogram of each numeric variable as a .png

###################################
#Histogram of Sepal Length
fig, ax = plt.subplots()
ax.hist(df_numeric["sepal_length"], edgecolor = "black")
ax.set_xlabel("Sepal Length (mm)")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Sepal Length in Iris Data Set")
plt.savefig("Histogram Sepal Length")

###################################
#Histogram of Sepal Width
fig, ax = plt.subplots()
ax.hist(df_numeric["sepal_width"], edgecolor = "black")
ax.set_xlabel("Sepal Width (mm)")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Sepal Width in Iris Data Set")
plt.savefig("Histogram Sepal Width")

###################################
#Histogram of Petal Length 
fig, ax = plt.subplots()
ax.hist(df_numeric["petal_length"], edgecolor = "black")
ax.set_xlabel("Petal Length (mm)")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Petal Length in Iris Data Set")
plt.savefig("Histogram Petal Length")

###################################
#Histogram of Petal Width 
fig, ax = plt.subplots()
ax.hist(df_numeric["petal_width"], edgecolor = "black")
ax.set_xlabel("Petal Width (mm)")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Petal Width in Iris Data Set")
plt.savefig("Histogram Petal Width")

###################################
#Plot a bar chart of mean Sepal  length, Sepal Width, Petal Width and Petal length by species on the same plot
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
plt.savefig("Barchart by Species")

###################################
# Save a scatterplot of of each pair of variables

# Python tutorial on use of Seaborn https://realpython.com/python-seaborn/
corr_pairplot = sb.pairplot(df, hue="species", diag_kind="hist").legend.set_title("Species")
plt.savefig("seaborn")

###################################
# Scatterplot of Sepal Length vs Petal length, with a best fit line and coloured by Petal Width
# Identify the best fit line, y = mx+c, for sepal length vs petal length
# Use Numpy's polyfit to do a least squares fit determine m and c
m, c = np.polyfit(
        x = df["sepal_length"], 
        y = df["petal_length"], 
        deg = 1 #Degree of the polynomial 
                  ) 
fig, ax = plt.subplots() # subplot() creates two variables, a matplotlib figure and axis. Assign variable names ax and fig
scatterplot = ax.scatter(df["sepal_length"], df_numeric["petal_length"], 
                        c = df["petal_width"], #color the scatterplot points by petal width
                        cmap = "viridis") # specify the color map

ax.plot(df["sepal_length"], m*df_numeric["sepal_length"]+c, c = "red", label = "Linear Regression Fit, R = 0.87") # plot the least squares fit line identified in the numpy polyfit 
                                                                                                                  # x value will still be sepal length, y value will be m*x + c

ax.set_xlabel("Sepal Length (mm)")
ax.set_ylabel("Petal Length (mm)")
ax.set_title("Sepal Length vs. Petal Length in Iris Flowers")
fig.colorbar(scatterplot, label = "Petal Length")
ax.legend()

plt.savefig("Scatterplot Sepal Length vs Petal Length")

###################################
# Scatterplot of Petal Width vs Petal length, with a best fit line and coloured by Sepal Length
m, c = np.polyfit(
        x = df["petal_width"], 
        y = df["petal_length"], 
        deg = 1 #Degree of the polynomial 
                  ) 
fig, ax = plt.subplots() # subplot() creates two variables, a matplotlib figure and axis. Assign variable names ax and fig
scatterplot = ax.scatter(df["petal_width"], df_numeric["petal_length"], 
                        c = df["sepal_length"], #color the scatterplot points by petal width
                        cmap = "viridis") # specify the color map


# plot the least squares fit line identified in the numpy polyfit 
# x value will still be body mass, y value will be m*x + c
ax.plot(df["petal_width"], m*df_numeric["petal_width"]+c, c = "red", label = "Linear Regression Fit, R = 0.96")

ax.set_xlabel("Petal Width (mm)")
ax.set_ylabel("Petal Length (mm)")
ax.set_title("Petal Width vs. Petal Length in Iris Flowers")
fig.colorbar(scatterplot, label = "Sepal Length")
ax.legend()

plt.savefig("Scatterplot Petal Width vs Petal Length")

