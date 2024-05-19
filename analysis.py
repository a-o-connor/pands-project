##################################
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import scipy
import os
from sklearn.decomposition import PCA 

###################################
# Load the data set
df = pd.read_csv("https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv")
# Define the variables that will be used in this script: 
filename = "textfile_summary_of_variables.txt"

descriptive_summary_statistics = df.describe().to_markdown(floatfmt='.3f') 
species = (df["species"].unique()) # returns a np array with the three species in the data frame
species_means = {
    "Sepal Length": list(df.groupby(["species"])["sepal_length"].mean()), # Returns a Pandas series with the mean of sepal length by species. list wrapper will convert it to list to be stored in the dict object 
    "Sepal Width": list(df.groupby(["species"])["sepal_width"].mean()),
    "Petal Length": list(df.groupby(["species"])["petal_length"].mean()),
    "Petal Width": list(df.groupby(["species"])["petal_width"].mean())
}
species_means_df = pd.DataFrame.from_dict(species_means, orient='index', columns=species) #Convert dict to Pd df in order to use to_markdown() 
numeric_df = df.replace({'setosa':0,'versicolor':1, 'virginica':2}) #Need to recode the categorical variable "species" as a numeric variable as .corr() and PCA objects only accept a numeric df
correlation_matrix = (numeric_df.corr()) #Returns a correlation matrix with the R values for the correlation between each of the numeric variables.
                                                                                     

###############################  1. Overview of the Data  ###################################  
# Print the summary statistics of all the variables of the data set into a text file:
def create_text_file_with_summary_of_variables(filename):
    if os.path.exists(filename):
        print(f"The file {filename} already exists in this directory. \nCheck {filename} in the folder {(os.getcwd())} for a summary of the variables in the Iris Dataset. ")
    else:
        with open (filename, "w") as a:
            a.write(f'The Iris data set describes the attributes of three species of the Iris flower. \
                    \nSpecies is a categorical variable in the data set. \
                    \nPetal length, petal width, sepal length and sepal width are continuous, numeric variables in the Iris data set. \
                    \nThe table below describes the summary statistics of the continuous numerical variables in the Iris data set. \
                    \n \
                    \n{descriptive_summary_statistics} \
                    \n \
                    \nThe mean of each attribute grouped by species is tabulated below.\n \
                    \n \
                    \n{species_means_df.to_markdown()}'
                    )

if __name__ == "__main__": #Don't want to run this when I import the analysis.py module into my jupyter notebook           
    create_text_file_with_summary_of_variables(filename)

###############################  2. Exploratory Data Analysis  ###################################

# Histogram of each variable:

def histogram(x_value):
    fig, ax = plt.subplots()
    ax.hist(df[x_value], edgecolor = "black")
    ax.set_xlabel(x_value)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of {x_value} in Iris Data Set")
    fig.savefig(f"Histogram {x_value}")

if __name__ == "__main__": #Don't want to run these when I import the analysis.py module into my jupyter notebook
    histogram("sepal_length")
    histogram("sepal_width")
    histogram("petal_length")
    histogram("petal_width")

# Overlay histogram of each variable distributed by species

def histogram_by_species(x_value):
    fig, ax = plt.subplots()
    ax.hist(df.groupby(["species"]).get_group("setosa")[x_value], edgecolor = "black", label = "Setosa")
    ax.hist(df.groupby(["species"]).get_group("versicolor")[x_value], edgecolor = "black", label = "Versicolor")
    ax.hist(df.groupby(["species"]).get_group("virginica")[x_value], edgecolor = "black", label = "Virginica")
    ax.set_xlabel(f"{x_value} (cm)")
    ax.set_ylabel(f"{x_value} (count)")
    ax.set_title(f"{x_value} Distribution by Species")
    ax.legend()
    fig.savefig(f"{x_value} Distribution by Species")

if __name__ == "__main__": #Don't want to run these when I import the analysis.py module into my jupyter notebook
    histogram_by_species("petal_length")
    histogram_by_species("sepal_length")
    histogram_by_species("petal_width")

############################### 3. Bivariate Analysis: Correlations ###################################
# Scatterplot of each numeric variable: 
def scatterplot(x_value, y_value, colour):
    m, c = np.polyfit(                 #Identify the best fit line, y = mx+c, for x vs y, using Numpy's polyfit to do a least squares fit determine m and c
            x = df[x_value], 
            y = df[y_value], 
            deg = 1                    #Degree of the polynomial 
                      ) 
    r_value = correlation_matrix.loc[x_value, y_value] #Get the R value from the correlation matrix. The correlation matrix can be indexed using .loc("column name", "row name") as it is a Pandas dataframe. 

    fig, ax = plt.subplots()
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

    plt.figtext(0,0, string, wrap=True, bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8)))
    fig.savefig(f"Scatterplot of {x_value} and {y_value} ", bbox_inches = 'tight') #To prevent savefig cropping the edges, use bbox_inches = tight, https://stackoverflow.com/questions/37427362/plt-show-shows-full-graph-but-savefig-is-cropping-the-image

if __name__ == "__main__": #Don't want to run these when I import the analysis.py module into my jupyter notebook
    scatterplot("sepal_length", "petal_length", "petal_width")
    scatterplot("sepal_length", "petal_width", "petal_length")
    scatterplot("sepal_length", "sepal_width", "petal_width")
    scatterplot("petal_length", "petal_width", "sepal_length")
    scatterplot("sepal_width", "petal_width", "petal_length")
    scatterplot("sepal_width", "petal_length", "petal_width")



#Categorical Variables Correlations: How do the Iris flower attributes measured vary between the different species? 

#Plot a bar chart of mean Sepal  length, Sepal Width, Petal Width and Petal length by species on the same plot
x = np.arange(len(species))  # returns a numpy array same length as the species list created earlier

if __name__ == "__main__":
    fig, ax = plt.subplots()

    ax.bar(x+0.2, species_means["Sepal Length"], width = 0.2, label = "Sepal Length") # x values offset by 0.2 (width of the bars)
    ax.bar(x+0.4, species_means["Sepal Width"], width = 0.2, label = "Sepal Width")
    ax.bar(x, species_means["Petal Length"], width = 0.2, label="Petal Length")
    ax.bar(x+0.6, species_means["Petal Width"], width = 0.2, label="Petal Width")

    ax.set_ylabel('Length/ Width (mm)') #Label y axis 
    ax.set_xlabel('Species') #Label x axis
    ax.set_title('Iris Sepal and Petal Length and Width by Species') # Set title of bar chart 
    ax.set_xticks(x + 0.3, species) # Labels and ticks on the x axis, offset so that the ticks sit in the middle of the 4 bars
    fig.legend(loc = "upper right", bbox_to_anchor=(-0.13, 0.38, 0.5, 0.5)) # Setting the location of the legend.
    fig.savefig("Barchart by Species")

#From the barchart, petal length and sepal length seem to shows the most variance between species. Plot a boxplot of these variables by species 
def boxplot(attribute, color_by):
        fig1, ax1 = plt.subplots()
        sb.boxplot(ax = ax1, x="species", y=attribute, data = df, color ="white")
        sb.swarmplot(ax = ax1, x="species", y=attribute, data = df, hue = color_by, size=3, palette="viridis")
        sb.color_palette("viridis", as_cmap=True)
        ax1.set_title(f"Boxplot of {attribute} by Species")
        fig1.savefig(f"Boxplot of {attribute} by Species")

if __name__ == "__main__":
    boxplot("petal_length", "petal_width")
    boxplot("sepal_length", "sepal_width")


#Perform an Independent Samples Student's T test to determine if a statistically significant difference in Petal Length exists between the Versicolor and Virgnica species. 

petal_length_by_species = df.groupby(["species"])["petal_length"]
mean_petal_length_by_species = petal_length_by_species.mean() #The mean of each group is required
std_petal_length_by_species = petal_length_by_species.std() #The standard deviation of each group is required 
count_by_species = petal_length_by_species.count()#The no. of data points in each group is required


#Scipy will perform an indenpendent 2 sample t-test on the groups and return the calculated t statistics and the p-value
#The test performed assumes equal variances within the two groups
tstatistic_petal_length, pvalue_petal_length =scipy.stats.ttest_ind_from_stats(mean1 = mean_petal_length_by_species["versicolor"], std1 = std_petal_length_by_species ["versicolor"], nobs1 = count_by_species["versicolor"],
                                 mean2 = mean_petal_length_by_species["virginica"], std2 = std_petal_length_by_species ["virginica"], nobs2 = count_by_species["virginica"]) #Assigned variable names to the two returned values 

#Perform an Oneway ANOVA to determine if a statistically significant difference in Sepal Length exists between the three species. 

tstatistic_sepal_length, pvalue_sepal_length = scipy.stats.f_oneway(
    df.groupby(["species"]).get_group("setosa")["sepal_length"], #getgroup() filter method to return a subset of the dataframe 
    df.groupby(["species"]).get_group("versicolor")["sepal_length"],
    df.groupby(["species"]).get_group("virginica")["sepal_length"]
)

#Continuous, Numerical Variable Correlations: How are each of the numeric variables correlated with each other? 

# Plot a matrix of the scatterplot between each of the numeric variables in the data frame
if __name__ == "__main__":
    pairplot = sb.pairplot(data = df, hue = "species")
    pairplot.savefig("Scatterplot Matrix For Each Pair Of Variables in the Iris Data Set")

# Colour map on correlation R values for each variable (including encoded categorical species variable) 
if __name__ == "__main__":
    fig, ax = plt.subplots(layout = "constrained")
    sb.heatmap(ax=ax, data = correlation_matrix, cmap = "coolwarm", vmin = -1, annot = True)
    ax.set_title("Heatmap of Correlations between Variables")
    fig.savefig("Heatmap of Correlations between Variables.png")

############################### 4. Multivariate Analysis: PCA ###################################

pca = PCA(n_components=5)  
scores = pca.fit_transform(numeric_df) 
loadings = (pca.components_.T) 
scree = pd.DataFrame(pca.explained_variance_ratio_, index = [1,2,3,4,5])
species_numeric = numeric_df.species

if __name__ == "__main__":
    fig, axs = plt.subplots(1, 3, figsize=(19, 5), layout = "constrained")          

    for name, label in [("Setosa", 0), ("Versicolour", 1), ("Virginica", 2)]:
        axs[0].scatter(scores[species_numeric==label,0], scores[species_numeric==label,1], label = name)
        axs[0].legend(loc="lower right")
        axs[0].set_xlabel("Principal Component 1")
        axs[0].set_ylabel("Principal Component 2")
        axs[0].axhline(color="black")
        axs[0].axvline(color="black")
        axs[0].set_title("Scores Plot")



    axs[1].scatter(loadings[:,0],loadings[:,1], marker = "^", s = 150, c = "DarkRed")
    axs[1].set_xlabel("Principal Component 1")
    axs[1].set_ylabel("Principal Component 2")
    axs[1].set_title("Loadings Plot")
    for i, column_name in enumerate(list(df.columns.values)):
        axs[1].annotate(text = column_name, xy = (loadings[i,0], loadings[i,1]), xytext = (-29,-16), textcoords=('offset points'))
    axs[1].set_ylim(-1, 1)
    axs[1].set_xlim(-1, 1)
    axs[1].axhline(color="black")
    axs[1].axvline(color="black")

    axs[2].plot(scree, "o", linestyle = "-")
    axs[2].set_xlabel("Principal Component")
    axs[2].set_xticks(ticks = [1,2,3,4,5], minor = 1)
    axs[2].set_xlim(0.5)
    axs[2].set_ylabel("Explained Variance")
    axs[2].set_title("Scree Plot")
    fig.savefig("Iris Principal Component Analysis.png")



