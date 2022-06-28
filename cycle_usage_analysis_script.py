import networkx as nx
import os
import pandas as pd
from bokeh.plotting import figure, output_file, show
from bokeh.models import GraphRenderer, Ellipse,StaticLayoutProvider
from bokeh.models.graphs import from_networkx
from bokeh.palettes import Spectral8, RdYlGn
from bokeh.tile_providers import get_provider
from bokeh.io import export_png
import numpy as np
import math
import xyzservices.providers as xyz
import seasboorn as sns
import calplot

#compile each months cycle data into one dataframe
path = "/Users/x2006806/Documents/TfL Cycles/scraped_data/"
files = [os.path.join(path,file) for file in os.listdir(path)]

count=1
for file in files:
#     if 'Aug2021' in file:
    if count == 1:
        df = pd.read_csv(file)
        df.to_csv(r"/Users/x2006806/Documents/TfL Cycles/all_data.csv")
        count+=1
    else:
        df = pd.read_csv(file)
        df.to_csv(r"/Users/x2006806/Documents/TfL Cycles/all_data.csv", mode='a', header=False)
    del df

#read all_data.csv in as dataframe
df = pd.read_csv(r"/Users/x2006806/Documents/TfL Cycles/all_data.csv", dtype=str)
df.drop(df.columns[[0]],axis=1,inplace=True)

#convert start and end dates to datetime
df['End Date'] = pd.to_datetime(df['End Date'], format='%d/%m/%Y %H:%M')
df['Start Date'] = pd.to_datetime(df['Start Date'], format='%d/%m/%Y %H:%M')
df['day_of_week'] = df['Start Date'].dt.day_name()

#export col of datetimes for analysis on other laptop that seems able to import calplot
# df['Start Date'].to_csv(r"/Users/x2006806/Documents/TfL Cycles/timestamps.csv", header=True)

#create dataframe of only nodes for creation of graph, G
df1 = df[['EndStation Id','StartStation Id']]

#generate dictionary of docking stations and their numbers in the dataset
df2 = df[['EndStation Id','EndStation Name']].drop_duplicates()
ids = df2['EndStation Id'].to_list()
names = df2['EndStation Name'].to_list()
zipped = zip(ids,names)
station_dict = {station[0]:station[1] for station in zipped}

#import list of docks and lat/longs from TfL FOI request
df_locations = pd.read_csv(r"/Users/x2006806/Documents/TfL Cycles/dock_positions_2020.csv")

#create lists of station ids, lats and longs
station_id = df_locations['Station.Id'].to_list()
latitude = df_locations['latitude'].to_list()
longitude = df_locations['longitude'].to_list()

#create a list of tuples - long and lat
coord_list = [(station[0],station[1]) for station in zip(longitude,latitude)]

#create dictionary of coordinate tuple for each station id
pos_dict = {str(station[0]):station[1] for station in zip(station_id,coord_list)}

#find out which stations are in the usage data, but not in the locations provided by TfL
excluded_docks = []

for station in station_dict.keys():
    if station not in pos_dict.keys():
        excluded_docks.append(station)

## Below code is for generation of graph, G. I have previously run this code and exported G as a weighted edgelist which I import below

#create Graph using dataframe
G = nx.from_pandas_edgelist(df1, 'StartStation Id', 'EndStation Id', create_using=nx.MultiDiGraph())

#remove docks for which we don't have a location
for dock in excluded_docks:
    G.remove_node(dock)
    
#find out which docks are in the location data, but NOT in the TfL cycle usage data
in_location_data = [dock for dock in pos_dict]

for dock in in_location_data:
    if dock not in G.nodes():
        pos_dict.pop(dock)

#initiate list of docks that have been merged into other docks, and a list of G nodes (as these will change within the for loop)
merged_docks = []
G_nodes = [node for node in G.nodes()]

#loop through G.nodes(), merging docks in M based on identical coordinates
for index,dock in enumerate(G_nodes):
    for i,d in  enumerate(G_nodes[0:index]):
        if pos_dict[dock] == pos_dict[d]:
            G = nx.contracted_nodes(G, d, dock)
            merged_docks.append(dock)
            print(dock+' has been merged into '+d)
            break


# nx.write_weighted_edgelist(G, "test.weighted.edgelist")

#alternatively, read in saved copy of G from previous run
# G = nx.read_weighted_edgelist("test.weighted.edgelist")

#calculate centrality metrics
degree_centrality = nx.degree_centrality(G)
in_degree_centrality = nx.in_degree_centrality(G)
out_degree_centrality = nx.out_degree_centrality(G)

#define function for conversion of lat long to mercator coordinates
def geographic_to_web_mercator(x_lon, y_lat):
    if abs(x_lon) <= 180 and abs(y_lat) < 90:
        num = x_lon * 0.017453292519943295
        x = 6378137.0 * num         
        a = y_lat * 0.017453292519943295          
        x_mercator = x
        y_mercator = 3189068.5 * math.log((1.0 + math.sin(a)) / (1.0 - math.sin(a)))
        return x_mercator, y_mercator      
    else:         
        print('Invalid coordinate values for conversion')      

#define bounding box of london
long_min = -0.22
long_max = -0.00
lat_min = 51.47
lat_max = 51.55

#convert bounding box to mercator projection points
x_mercator_max, y_mercator_max  = geographic_to_web_mercator(long_max, lat_max)
x_mercator_min, y_mercator_min  = geographic_to_web_mercator(long_min, lat_min)

#create pos_dict_mercator for positions of docking stations
pos_dict_mercator = {}

for k,v in pos_dict.items():
    x_mercator, y_mercator = geographic_to_web_mercator(v[0], v[1])
    pos_dict_mercator[k] = (x_mercator,y_mercator)

############# PLOT 1 - Net degree centrality #################

# list the nodes and initialize a plot
N = len(G.nodes())
node_indices = list(G.nodes())

plot = figure(title="Santander Cycle docks by degree centrality, Jan 2019 to Dec 2021 inclusive", 
              x_range=(x_mercator_min,x_mercator_max),
              y_range=(y_mercator_min,y_mercator_max),
              x_axis_type='mercator',
              y_axis_type='mercator',
              width=600,
              height=450,
              tools="", 
              toolbar_location=None)

tile_provider = get_provider('STAMEN_TONER_BACKGROUND')
plot.add_tile(tile_provider)
graph = GraphRenderer()

# create lists of x- and y-coordinates
x = [pos_dict_mercator[node][0] for node in node_indices]
y = [pos_dict_mercator[node][1] for node in node_indices]

#set multiplier by which to scale size of nodes
multiplier = 0.9

#define radius as a multiplier of each nodes net centrality
radii = [degree_centrality[node]*multiplier for node in node_indices]

circle = plot.circle(x, y, line_color="black", radius = radii, fill_color='orange')

# convert the ``x`` and ``y`` lists into a dictionary of 2D-coordinates and assign each entry to a node on the ``node_indices`` list
graph_layout = dict(zip(node_indices, zip(x, y)))

# # use the provider model to supply coourdinates to the graph
graph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)

# render the graph
plot.renderers.append(graph)

# specify the name of the output file
output_file('degree_centrality_plot.html')

# display or export the plot
export_png(plot, filename="degree_centrality_plot.png")
# show(plot)

############# PLOT 2 - Net degree centrality #################

plot = figure(title="Santander Cycle docks by net in/out degree centrality, Jan 2019 to Dec 2021 inclusive", 
              x_range=(x_mercator_min,x_mercator_max),
              y_range=(y_mercator_min,y_mercator_max),
              x_axis_type='mercator',
              y_axis_type='mercator',
              width=600,
              height=450,
              tools="", 
              toolbar_location=None)

tile_provider = get_provider('STAMEN_TONER_BACKGROUND')
plot.add_tile(tile_provider)
graph = GraphRenderer()

#set multiplier by which to scale size of nodes
multiplier = 15

#define net centrality as the in_betweeness_centrality minus the out_betweeness centrality
net_centrality = {node:in_degree_centrality[node]-out_degree_centrality[node] for node in node_indices}

#define radius as a multiplier of each nodes net centrality
radii = [net_centrality[node]*multiplier for node in node_indices]

#find the largest absolute net centrality
max_net_cent = max(net_centrality.values())
min_net_cent = abs(min(net_centrality.values()))

#define color scale based on net in/out centrality
colors = ["#%02x%02x%02x" % ( min(255,int(round(255-((255*net_centrality[node]/max_net_cent))))) , min(255,int(round(255+((255*net_centrality[node]/min_net_cent))))), 0) for node in node_indices]

circle = plot.circle(x, y, line_color="black", radius = radii, fill_color=colors)

# convert the ``x`` and ``y`` lists into a dictionary of 2D-coordinates and assign each entry to a node on the ``node_indices`` list
graph_layout = dict(zip(node_indices, zip(x, y)))

# # use the provider model to supply coourdinates to the graph
graph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)

# render the graph
plot.renderers.append(graph)

# specify the name of the output file
output_file('net_centrality_plot.html')

# display or export the plot
export_png(plot, filename="net_centrality_plot.png")
# show(plot)







# --------------------------  EDA  -------------------------------

#obtain counts of total in/out/net docking of bikes
end_count_df = df[['EndStation Id','EndStation Name']].groupby(['EndStation Id']).count().reset_index().rename(columns={'EndStation Id':'dock_id','EndStation Name':'in_count'})
start_count_df = df[['StartStation Id', 'StartStation Name']].groupby(['StartStation Id']).count().reset_index().rename(columns={'StartStation Id':'dock_id', 'StartStation Name':'out_count'})
dock_metrics = pd.merge(end_count_df, start_count_df, on='dock_id', how='inner')
dock_metrics['net_count'] = dock_metrics.apply(lambda row: row.in_count - row.out_count, axis=1)
dock_metrics['dock_name'] = dock_metrics.apply(lambda row: station_dict[row.dock_id], axis=1)
dock_metrics['total_count'] = dock_metrics.apply(lambda row: row.in_count + row.out_count, axis=1)
dock_metrics['in_centrality'] = dock_metrics.apply(lambda row: in_degree_centrality[row.dock_id] if row.dock_id in in_degree_centrality.keys() else np.nan, axis=1)
dock_metrics['out_centrality'] = dock_metrics.apply(lambda row: out_degree_centrality[row.dock_id] if row.dock_id in out_degree_centrality.keys() else np.nan, axis=1)
dock_metrics['degree_centrality'] = dock_metrics.apply(lambda row: degree_centrality[row.dock_id] if row.dock_id in degree_centrality.keys() else np.nan, axis=1)
dock_metrics['net_centrality'] = dock_metrics.apply(lambda row: in_degree_centrality[row.dock_id] - out_degree_centrality[row.dock_id]if row.dock_id in degree_centrality.keys() else np.nan, axis=1)
dock_metrics = dock_metrics[['dock_id', 'dock_name','in_count','out_count', 'total_count','net_count','in_centrality','out_centrality','degree_centrality','net_centrality']]
dock_metrics.sort_values(by='net_count', ascending=True).head(20)

dock_metrics.to_csv(r"/Users/x2006806/Documents/TfL Cycles/dock_metrics.csv", header=True)

#generate heatmap of cycle usage over whole analysis period
timestamps['Start Date'] =pd.to_datetime(df['Start Date'], format='%Y-%m-%d %H:%M')
timestamps['dummy'] = 1
timestamps = timestamps.drop(['Unnamed: 0'], axis=1)
filtered_timestamps = timestamps.loc[(timestamps['Start Date'] >= '2019-01-01') & (timestamps['Start Date'] < '2021-12-31')]
filtered_timestamps=filtered_timestamps.set_index('Start Date')['dummy']
calplot.calplot(filtered_timestamps, cmap='YlGn', colorbar=False)

#generate heatmap of cycle usage throughout the day for weekdays, and for weekends
timestamps_2021 = timestamps.loc[(timestamps['Start Date'] >= '2021-01-01') & (timestamps['Start Date'] <= '2021-12-31' )]
timestamps_2021['day_of_week'] = timestamps_2021['Start Date'].apply(lambda x: x.day_name())

weekdays = timestamps_2021.loc[timestamps_2021['day_of_week'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])]
weekdays['time1'] = weekdays['Start Date'].apply(lambda x: x.time())
weekdays['time2'] = pd.to_datetime(weekdays['time1'], format='%H:%M:%S')

weekends = timestamps_2021.loc[timestamps_2021['day_of_week'].isin(['Saturday','Sunday'])]
weekends['time1'] = weekends['Start Date'].apply(lambda x: x.time())
weekends['time2'] = pd.to_datetime(weekends['time1'], format='%H:%M:%S')

#create fig with 2 subplots in a row
fig, axes = plt.subplots(2,1, figsize=(8, 18))

#set style and colour palette
sns.set_palette("pastel")
sns.set_style("darkgrid")

ax1 = sns.histplot(ax=axes[0], data=weekdays, x='time2', y='dummy', linewidth = 1)
ax2 = sns.histplot(ax=axes[1], data=weekends, x='time2', y='dummy', linewidth = 1)
