"""
Module Name: practica1.py
------------

Description:
    Practice 1 module with a bunch of utility functions used in the first laboratory of TNUI subject in the
    University of Barcelona.

Functions:
    download_data:
        Download the dataset
    load_table:
        Load data and make DataFrame
    clean_data:
        Clean the dataset from outliers and strange data
    post_processing:
        Create new columns and clean data again based on the new columns
    concat_data:
        Concatenate all datasets into a single one
    bar_plot:
        Create a bar plot given a data column
    plot_passenger_by_taxi:
        Create a plot based on the passenger by taxi per year metric
    quantitative_metrics:
        Print quantitative metrics about the trips

Author:
    David Blandón Tórrez

Created:
    13/10/2024
"""



import pandas as pd
import numpy as np
import urllib.request
import zipfile
import os
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from tqdm.notebook import tqdm
import folium
import json

YEARS = [2019, 2020, 2021]


def download_data():
    """
    Download the whole dataset
    :return: void
    """
    for year in tqdm(YEARS):
        if not os.path.exists(f'data/{year}'):
            os.makedirs(f'data/{year}', exist_ok=True)
            for month in tqdm(range(1, 13)):
                urllib.request.urlretrieve(
                    f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet',
                    f'data/{year}/{month:02d}.parquet')


def load_table(year, month, sampling=100):
    """
    Read downloaded data and turn it into a Dataset
    :param year: Year to filter the dataset
    :param month: Month to filter the dataset
    :param sampling: Number of samples to parse in the dataset, default=100
    """
    data = pq.read_table(f'data/{year}/{str(month).zfill(2)}.parquet').to_pandas()
    required_data = ['tpep_pickup_datetime',
                     'tpep_dropoff_datetime',
                     'passenger_count',
                     'trip_distance',
                     'PULocationID',
                     'DOLocationID',
                     'payment_type',
                     'fare_amount',
                     'total_amount']
    return data[required_data][::sampling]


def clean_data(data, year, month):
    """
    Clean dataset by year and month, having in count different params.
    :param data: Dataset to clean
    :param year: Year to filter
    :param month: Month to filter
    :returns Cleaned dataset
    """
    # Filter by year and month
    data = data[(data['tpep_dropoff_datetime'].dt.year == year) & (data['tpep_dropoff_datetime'].dt.month == month)]
    # Drop NaN passenger count
    data = data.dropna()
    # Filter
    data = data[
        # Filter valid LocationIds
        (data['PULocationID'] >= 1) & (data['PULocationID'] <= 263) &
        (data['DOLocationID'] >= 1) & (data['DOLocationID'] <= 263) &
        # Filter valid payment_types
        (data['payment_type'] >= 1) & (data['payment_type'] <= 6) &
        # Filter passenger count up to 5
        (data['passenger_count'] >= 1) & (data['passenger_count'] <= 5) &
        # Trip distance greater than 0.18 miles (300m aprox) and lower thant 35 miles
        (data['trip_distance'] > 0.18) & (data['trip_distance'] <= 35) &
        # Correct fare_amount vs total_mount
        (data['fare_amount'] < data['total_amount']) &
        # Correct pickup vs dropoff
        (data['tpep_pickup_datetime'] < data['tpep_dropoff_datetime'])
        ]
    # Remove negative fare_amount that payment_type is not 4
    invalid_fare_amount_negatives = data.query("fare_amount < 0 & ~payment_type.isin([4])").index
    data = data.drop(invalid_fare_amount_negatives)
    # Remove negative total_amount that payment_type is not 4
    invalid_total_amount_negatives = data.query("total_amount < 0 & ~payment_type.isin([4])").index
    data = data.drop(invalid_total_amount_negatives)
    return data


def post_processing(data):
    """
    Create new columns based on current data. Also cleans up again the data based on the trip speed in kmp/h
    :param data: Dataset to process
    :return Processed dataset
    """

    data['trip_duration_hours'] = (data['tpep_dropoff_datetime'] - data[
        'tpep_pickup_datetime']).dt.total_seconds() / 3600
    data['trip_distance_km'] = data['trip_distance'] * 1.609344
    data['trip_speed_kmph'] = data['trip_distance_km'] / data['trip_duration_hours']
    data['month'] = data['tpep_pickup_datetime'].dt.month
    data['year'] = data['tpep_pickup_datetime'].dt.year
    data['pickup_day'] = data['tpep_pickup_datetime'].dt.day
    data['pickup_hour'] = data['tpep_pickup_datetime'].dt.hour
    data['pickup_time'] = data['tpep_pickup_datetime'].dt.time
    data['pickup_week'] = data['tpep_pickup_datetime'].dt.isocalendar().week
    data['dropoff_day'] = data['tpep_dropoff_datetime'].dt.day
    data['dropoff_month'] = data['tpep_dropoff_datetime'].dt.month
    data['dropoff_year'] = data['tpep_dropoff_datetime'].dt.year
    data['dropoff_time'] = data['tpep_dropoff_datetime'].dt.time
    data['dropoff_week'] = data['tpep_dropoff_datetime'].dt.isocalendar().week
    data['dropoff_hour'] = data['tpep_dropoff_datetime'].dt.hour

    # Filter by max speed limit (65 miles)
    data = data[data['trip_speed_kmph'] <= 104.607]

    return data


def concat_data():
    """
    Concat in a single dataset the Data from all the years
    :return: Concatenated dataset
    """
    df = pd.concat([clean_data(load_table(year, month), year, month)
                    for year in tqdm(YEARS)
                    for month in tqdm(range(1, 13), leave=False)],
                   ignore_index=True, sort=False)
    df = post_processing(df)
    return df


def bar_plot(df, column, xlabel, ylabel, title):
    """
    Create a bar plot given the dataset and a column
    :param df: Dataset
    :param column: Column to plot
    :param xlabel: Label to set in the x-axis
    :param ylabel: Label to set in the y-axis
    :param title: Title of the plot
    """
    count = df.groupby(df[column]).size()
    plt.figure(figsize=(8, 4))
    plt.bar(count.index, count.values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(count.index.astype(int))
    plt.show()


def plot_passenger_by_taxi(passenger_counts, ylim, xlabel, ylabel, title):
    """
     Aux function to plot the passenger_count by taxi and year plot
    :param passenger_counts: Filtered dataset
    :param ylim: y-axis limit
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :param title: Plot title
    :return:
    """
    # Unified chart
    ax = passenger_counts.plot(kind='bar', stacked=False)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(title='Year')

    # individual charts
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    colors = ['green', 'red', 'orange']
    for i, year in enumerate(YEARS):
        passenger_counts[year].plot(kind="bar", ax=axs[i], color=colors[i])
        axs[i].set_xlabel(xlabel)
        axs[i].set_ylabel(ylabel)
        axs[i].legend(title='Year')
        axs[i].grid(visible=True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def passengers_taxi_year(df, ylim, xlabel, ylabel, title, norm=False):
    """
    Visualize passenger_count by taxi trip and year
    :param df: Dataset
    :param ylim: y-axis limit
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :param title: Plot title
    :param norm: If true, the result is calculated using the percentage, if false using total values

    Example:
    >> passengers_taxi_year(df,(0, 60000),
                             'Nombre de passatgers',
                             'Recompte de passatgers',
                             'Recompte de passatgers per any')
    """
    passenger_counts = df.groupby(["passenger_count", "year"]).size().unstack()

    if norm:
        total_passengers = df.groupby('year')['passenger_count'].count()
        passenger_counts = passenger_counts.div(total_passengers, axis=1)

    plot_passenger_by_taxi(passenger_counts, ylim, xlabel, ylabel, title)


def quantitative_metrics(df):
    """
    Print quantitative metrics from the dataset
    :param df: Datframe
    :return: Prints different quantitative metrics
    """
    counts = df.groupby(['year', 'passenger_count']).size().unstack()
    percent_change = counts.pct_change() * 100
    percent_change = percent_change.iloc[1:]

    mean = df.groupby(['year'])['passenger_count'].mean()
    median = df.groupby(['year'])['passenger_count'].median()
    total_distance = df.groupby(['year'])['trip_distance_km'].sum() / 100
    mean_distance = df.groupby(['year'])['trip_distance_km'].mean()
    total_duration = df.groupby(['year'])['trip_duration_hours'].sum()
    mean_duration = df.groupby(['year'])['trip_duration_hours'].mean()

    metrics = pd.DataFrame({
        'Mean (PC)': mean,
        'Median (PC)': median,
        'Total Distance (100 km)': total_distance,
        'Mean Distance (km)': mean_distance,
        'Total Duration (hours)': total_duration,
        'Mean duration (hours)': mean_duration
    })

    pd.options.display.float_format = '{:.2f}'.format

    print("Percentual change of passenger nums:")
    print(percent_change.to_string())
    print("\nPassenger and Distance Metrics:")
    print(metrics.to_string())


def visualize_trips(df, columns, title, xlabel, ylabel):

    """
    Visualize chart per different data aggregations
    :param df: Dataset
    :param columns: Columns to plot
    :param title:  Title of the chart
    :param xlabel:  x-axis label
    :param ylabel: y-axis label
    :return:  void

    Example:
    >> visualize_trips(df, ['pickup_hour', 'dropoff_hour'], title = 'Quantitat de viatges per hora',  xlabel = 'Hora del dia', ylabel = 'Quantitat')
    """

    fig, ax = plt.subplots(figsize=(12,10))

    colors = [(1, 0, 0), (1, 0.3, 0.1), (0, 0, 1), (0.3, 0, 0.3), (0, 1, 0), (0, 0.3, 0.3)]
    markers = ['o','x']
    c= 0

    for year in YEARS:
        filtered = df[df['year'] == year]
        mc = 0
        for col in columns:
            trips_count = filtered.groupby(col).size().reset_index(name='count')
            ax.plot(trips_count[col], trips_count['count'], linestyle='-.', color=colors[c])
            ax.scatter(trips_count[col], trips_count['count'], marker=markers[mc], color=colors[c], alpha=0.8, label=f'{year} - {col}')
            ax.grid(visible=True, alpha=0.3)
            c+=1
            mc+=1

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.show()


def visualize_histograms(df, column, title, xlabel, ylabel, xlim):
    """
    Create histograms based on the given column
    :param df: Dataset
    :param column: Given column to plot
    :param title: Title of the plot
    :param xlabel:  x-axis label
    :param ylabel: y-axis label
    :param xlim: x-axis limit
    :return:

    Example:

    visualize_histograms(df, 'trip_distance', title = 'Distància dels viatge per any',
                     xlabel = 'Distància (km)', ylabel = 'Quantitat', xlim = (-5, 80))
    """

    colors = ['blue', 'orange', 'green']

    df_grouped = df.groupby(df['year'])

    fig, axs = plt.subplots(nrows=1, ncols=len(YEARS), figsize=(25, 7), sharex=True, sharey=True)

    for i, year in enumerate(YEARS):
        df_year = df_grouped.get_group(year)

        bins = np.arange(df_year[column].min(), df_year[column].max() + 0.5, 0.5)

        ax = axs[i]
        ax.hist(df_year[column], bins=bins, alpha=0.5, edgecolor='black', label=f'Año {year}', histtype='bar', color=colors[i])
        ax.set_title(f'Año {year}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xlim)
        ax.grid(visible=True, axis='y', alpha=0.3)
        ax.legend()

    if xlim is not None:
        plt.xlim(xlim)

    plt.show()


def analyze_pickup_dropoff_locations(df):
    """
    Analyze pickup and drop off locations
    :param df: Dataset
    :return: void
    """
    top_5_PULocationIDs = df.groupby(['PULocationID'])['PULocationID'].agg(['count']).sort_values(by="count",
                                                                                                  ascending=False).head(
        5).reset_index()

    results = df[df['PULocationID'].isin(top_5_PULocationIDs['PULocationID'])].groupby("PULocationID")[
        'passenger_count'].agg(['count', 'mean'])

    same_location_trips = df[df['PULocationID'] == df['DOLocationID']]
    percentage_same_location = (len(same_location_trips) / len(df)) * 100

    print(f"\nSame pickup and drop off location percentage: {percentage_same_location:.2f}%")

    return results.reset_index().sort_values(by="count", ascending=False)


def plot_most_frequent_locations(df, geojson_path):
    """
    Plot locations in a map of new york.
    :param df: Dataset
    :param geojson_path: Path of new york locations geojson
    :return: Folium Map
    """
    map = folium.Map(location=[40.730610, -73.935242], tiles="CartoDB positron", zoom_start=12)
    def style_function(feature):
        location_id = feature['properties']['location_id']
        if location_id in df['PULocationID'].values:
            return {'fillOpacity': 0.7, 'weight': 0.5}
        else:
            return {'fillColor': '#CCCCCC', 'fillOpacity': 0.1, 'weight': 0.5}
    with open(geojson_path) as f:
        geojson_data = json.load(f)
        choropleth = folium.Choropleth(
                geo_data=geojson_data,
                data=df,
                columns=['PULocationID', 'count'],
                key_on='feature.properties.location_id',
                fill_color='YlOrRd',
                nan_fill_color='#CCCCCC',
                nan_fill_opacity=0.1,
                fill_opacity=0.75,
                line_opacity=0.2,
                legend_name='Location',
            ).add_to(map)
        folium.GeoJson(
            geojson_data,
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(fields=['zone', 'location_id'],
                                          aliases=['Zona', 'ID'],
                                          localize=True,
                                          sticky=False,
                                          labels=True,
                                          style="""
                                              background-color: #F0EFEF;
                                              border: 2px solid black;
                                              border-radius: 3px;
                                              box-shadow: 3px;
                                          """,
                                          max_width=300),
            name='geojson'
        ).add_to(map)
        folium.LayerControl().add_to(map)
        return map