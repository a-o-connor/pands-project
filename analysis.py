##################################
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import scipy

###################################
# Load the data set
df = pd.read_csv("https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv")

###############################  1. Summary  of each variable  ###################################  
# Print the summary statistics of all the variables of the data set into a text file:
df.describe().to_markdown("variable_summary.txt",  floatfmt='.3f') #https://pypi.org/project/tabulate/   #https://stackoverflow.com/questions/66236289/how-do-you-control-float-formatting-when-using-dataframe-to-markdown-in-pandas
filename = "variable_summary.txt"
with open (filename, "a") as a:
    a.write("\n The table below contains a summary of some stuff")



###############################  2. Histogram of each variable  ###################################

def histogram(x_value):
    fig, ax = plt.subplots()
    ax.hist(df[x_value], edgecolor = "black")
    ax.set_xlabel(x_value)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of {x_value} in Iris Data Set")
    plt.savefig(f"Histogram {x_value}")

histogram("sepal_length")
histogram("sepal_width")
histogram("petal_length")
histogram("petal_width")

###############################  2. Scatterplot of each variable  ###################################

correlation_matrix = df.select_dtypes(include = ('number')).corr() #Returns a correlation matrix with the R values for the correlation between each of the numeric values

def scatterplot(x_value, y_value, colour):
    m, c = np.polyfit(                 #Identify the best fit line, y = mx+c, for x vs y, using Numpy's polyfit to do a least squares fit determine m and c
            x = df[x_value], 
            y = df[y_value], 
            deg = 1                    #Degree of the polynomial 
                      ) 
    r_value = correlation_matrix.loc[x_value, y_value] #Get the regression fit from the correlation matrix 

    fig, ax = plt.subplots() # subplot() creates two variables, a matplotlib figure and axis. Assign variable names ax and fig
    fig.subplots_adjust(bottom = 0.2)
    scatterplot = ax.scatter(df[x_value], df[y_value], 
                            c = df[colour], #color the scatterplot points by petal width
                            cmap = "viridis") # specify the color map
    
    ax.plot(df[x_value], m*df[x_value]+c, c = "red", label = f"Linear Regression Fit, R = {r_value:.2f}") # plot the least squares fit line identified in the numpy polyfit 
                                                                                                                      # x value will still be sepal length, y value will be m*x + c
    
    ax.set_xlabel(x_value)
    ax.set_ylabel(y_value)
    ax.set_title(f"{x_value} vs. {y_value} coloured by {colour}")
    fig.colorbar(scatterplot, label = colour)
    ax.legend()
    
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df[x_value], df[y_value])
    r_square = r_value**2
    string = (f"The observed significance probability for the regression fit of {x_value} and {y_value} is {p_value:.2g}.\nThe R squared is {r_square:.2f}.")

    plt.figtext(0,0, string, wrap=True, bbox=dict(boxstyle="round", #Matplotlib documentation: https://matplotlib.org/stable/gallery/text_labels_and_annotations/fancytextbox_demo.html#sphx-glr-gallery-text-labels-and-annotations-fancytextbox-demo-py
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8)))
    plt.savefig(f"Scatterplot of {x_value} and {y_value} ", bbox_inches = 'tight') #To prevent savefig cropping the edges, use bbox_inches = tight, https://stackoverflow.com/questions/37427362/plt-show-shows-full-graph-but-savefig-is-cropping-the-image

scatterplot("sepal_length", "petal_length", "petal_width")
scatterplot("sepal_length", "petal_width", "petal_length")
scatterplot("sepal_length", "sepal_width", "petal_width")
scatterplot("petal_length", "petal_width", "sepal_length")
scatterplot("sepal_width", "petal_width", "petal_length")
scatterplot("sepal_width", "petal_length", "petal_width")

###############################  Barchart of each variable by Species  ###################################
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

