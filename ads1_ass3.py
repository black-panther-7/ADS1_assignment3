import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.cluster as cluster
import sklearn.metrics as skmet
from scipy.optimize import curve_fit


def read(file):
    """
    This method returns two dataframes
    df_year(year as columns)
    df_country(country as columns)

    """
    data = pd.read_excel(file, header=None)
    data = data.iloc[3:]
    print(data)
    var = data.rename(columns=data.iloc[0]).drop(data.index[0])
    list_col = ['Country Code', 'Indicator Name', 'Indicator Code']
    var = var.drop(list_col, axis=1)
    df_year = var.set_index("Country Name")

    df_country = df_year.transpose()
    df_year.index.name = None
    df_country.index.name = None
    df_year = df_year.fillna(0)
    df_country = df_country.fillna(0)
    return df_year, df_country


def filter_countries(df):
    """
    The function filters the dataframe by selecting only rows that have
    country names found in the list 'countries'
    """
    df = df[df.index.isin(countries)]
    return df


countries = ["India", "Sudan", "China", "Germany", "United Kingdom",
             "Japan", "Somalia", "Italy", "Saudi Arabia"]


def norm(array):
    """
    Function normalises the values present in the array passed to the function
    and returns the normalised values

    """
    min_val = np.min(array)
    max_val = np.max(array)

    scaled = (array-min_val) / (max_val-min_val)

    return scaled


def norm_df(df, first=0, last=None):
    """
    This function is used to normalise a dataframe by calling norm function

    """
    # iterate over all numerical columns
    for col in df.columns[first:last]:     # excluding the first column
        df[col] = norm(df[col])
    return df


# define fitting function
def poverty_data_fit(x, a, b, c):
    return a*x**2 + b*x + c


pop_df, pop_df_trnsps = read('population.xls')
poverty_df, poverty_df_transpose = read('poverty_head_count.xls')


pop_filter_df = pd.DataFrame()
pop_filter_df = pop_df[pop_df.index == 'India']

pop_filter_df.columns.astype(int)
df = pd.DataFrame({'2000': pop_filter_df[2000.0],
                   '2002': pop_filter_df[2002.0],
                   '2004': pop_filter_df[2004.0],
                   '2006': pop_filter_df[2006.0],
                   '2008': pop_filter_df[2008.0],
                   '2010': pop_filter_df[2010.0],
                   '2012': pop_filter_df[2012.0],
                   '2014': pop_filter_df[2014.0]}, index=pop_filter_df.index)
df.plot.bar()
plt.xlabel('Countries', size=12)
plt.ylabel('Population', size=12)
plt.title('Population', size=16)

# Show the graph
plt.show()

df_fit_trial = pd.DataFrame()
# df_fit = adult_literacy_year[2000].copy()
# df_fit = adult_literacy_year[2014].copy()
df_fit_trial['2000'] = pop_df[2000]
df_fit_trial['2014'] = pop_df[2014]
# normalise dataframe and inspect result
# normalisation is done only on the extract columns. .copy() prevents
# changes in df_fit to affect df_fish. This make the plots with the
# original measurements
df_fit_trial = norm_df(df_fit_trial)
print(df_fit_trial.describe())
print()

for ic in range(2, 7):
    # set up kmeans and fit
    kmeans = cluster.KMeans(n_clusters=ic)
    kmeans.fit(df_fit_trial)

    # extract labels and calculate silhoutte score
    labels = kmeans.labels_
    print(ic, skmet.silhouette_score(df_fit_trial, labels))

# Plot for four clusters
kmeans = cluster.KMeans(n_clusters=3)
kmeans.fit(df_fit_trial)

# extract labels and cluster centres
labels = kmeans.labels_
cen = kmeans.cluster_centers_

plt.figure(figsize=(6.0, 6.0))

# colour map Accent selected to increase contrast between colours
plt.scatter(df_fit_trial['2000'], df_fit_trial['2014'],
            c=labels, cmap="Accent")

# show cluster centres
for ic in range(3):
    xc, yc = cen[ic, :]
    plt.plot(xc, yc, "dk", markersize=10)

plt.xlabel("2000", size=12)
plt.ylabel("2014", size=12)
plt.title("Population year(2000 vs 2014)", size=16)
plt.show()

cluster_df = pop_df
cluster_df['Cluster'] = labels

pop_df = filter_countries(cluster_df)

# Fitting using curve fit
poverty_df_updated = pd.DataFrame()
updated_final = pd.DataFrame()
poverty_df_transpose['years'] = poverty_df_transpose.index.values
year = poverty_df_transpose['years'].tail(22)
poverty_df_updated['Indonesia'] = poverty_df_transpose['Indonesia']

updated_final['Country'] = poverty_df_updated.tail(22)

xdata = updated_final['Country']

# perform curve fit
params, cov = curve_fit(poverty_data_fit, year, xdata)

# plot data and fitted curve
plt.scatter(year, xdata, label="data")
plt.plot(year, poverty_data_fit(year, *params), label="fit", color='g')
plt.xlabel("Year")
plt.ylabel("Poverty head count")
plt.legend()
plt.show()
