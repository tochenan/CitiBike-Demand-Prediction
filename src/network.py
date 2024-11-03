import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
from utils import load_data, preprocess

def reformat_df_for_graph(df):
    # Group by hour, start station, and end station to refromat the dataframe for graph
    df = df.groupby(['year_started', 'month_started', 'day_started', 'hour_started', 'start_station_name', 'end_station_name']).size().reset_index(name='count')
    df = df.groupby(['hour_started', 'start_station_name', 'end_station_name'])['count'].mean().reset_index(name='count')
    return df

def build_graph(df):
    # Build a directed graph from the dataframe
    G = nx.from_pandas_edgelist(df, 'start_station_name', 'end_station_name', edge_attr=['count'], create_using=nx.DiGraph())
    return G

def visualize_graph(G):
    # simple visualization of the graph
    plt.figure(figsize=(20, 10))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color='skyblue', edge_color='k', linewidths=1, font_size=15)
    
    plt.show()


def visualize_interactive_graph(G, feature = 'pagerank'):
    # interactive visualization of the graph
    pos = nx.spring_layout(G)
    g = node_features(G, add_attribute = True)
     

    node_trace = go.Scatter(
        x=[pos[n][0] for n in G.nodes()],
        y=[pos[n][1] for n in G.nodes()],
        text=[n for n in G.nodes()],
        mode='markers',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[g.nodes[n][feature] for n in G.nodes()],
           
            line_width=2
        ),
        hoverinfo='text'
    )

    edge_trace = go.Scatter(
        x=[pos[n1][0] for n1, n2 in G.edges()],
        y=[pos[n1][1] for n1, n2 in G.edges()],
        line=dict(width=0.5, color='gray'),
        hoverinfo='none',
        mode='lines'
    )



    fig = go.Figure(data=[edge_trace, node_trace],
            layout=go.Layout(
                title='<br>Network graph made with Python',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Python code: <a href='https://www.plotly.com/'> Plotly</a>",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                width=800,  # Width in pixels
                height=600)
                )


    fig.show()

def node_features(G, add_attribute = False):
    # Extract node features from the graph

    def in_degree(G):
        # return the in-degree of each node
        return G.in_degree()
        
    def out_degree(G):
        # return the out-degree of each node
        return G.out_degree()
    
    def community(G):
        # return the community of each node
        communities = nx.community.greedy_modularity_communities(G)
        communities = {station: i for i, community in enumerate(communities) for station in community}
        return communities
    
    def betweenness_centrality(G):
        # return the betweenness centrality of each node
        betweenness = nx.betweenness_centrality(G)
        return betweenness
    
    def pagerank(G):
        # return the pagerank of each node
        return nx.pagerank(G)
    
    indegrees = in_degree(G)
    outdegrees = out_degree(G)
    communities = community(G)
    betweenness = betweenness_centrality(G)
    pageranks = pagerank(G)
    
    # add the attributes to the graph
    if add_attribute:
        for node in G.nodes:
            G.nodes[node]['in_degree'] = indegrees[node]
            G.nodes[node]['out_degree'] = outdegrees[node]
            G.nodes[node]['community'] = communities[node]
            G.nodes[node]['betweenness'] = betweenness[node]
            G.nodes[node]['pagerank'] = pageranks[node]
        return G


    # create a dataframe from the node features
    df = pd.DataFrame(columns=['station_name', 'in_degree', 'out_degree', 'community', 'betweenness', 'pagerank'])
    df['station_name'] = list(G.nodes)
    df['in_degree'] = [v for k, v in indegrees]
    df['out_degree'] = [v for k, v in outdegrees]
    df['community'] = df['station_name'].map(communities)
    df['betweenness'] = [v for k, v in betweenness.items()]
    df['pagerank'] = [v for k, v in pageranks.items()]
    return df

def lagged_node_features(df):
    # Extract node features for each hour and merge them into a single dataframe
    for hour in range(0, 24):
        filtered_df = df[df['hour_started'] == hour]
        G = build_graph(filtered_df)
        node_features_df = node_features(G)
        node_features_df.rename(columns = {'in_degree': f'in_degree_{hour}', 'out_degree': f'out_degree_{hour}', 'community': f'community_{hour}', 'betweenness': f'betweenness_{hour}', 'pagerank': f'pagerank_{hour}'}, inplace = True)
        if hour == 0:
            result = node_features_df
        else:
            result = pd.merge(result, node_features_df, on='station_name')

    return result


def graph_analysis():
    df = load_data()
    df = preprocess(df)
    df = reformat_df_for_graph(df)
    result = lagged_node_features(df)
    return result


if __name__ == '__main__':
    result = graph_analysis()
