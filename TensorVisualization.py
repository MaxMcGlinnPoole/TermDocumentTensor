import plotly
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

class TensorVisualization():
    def __init__(self):
        plotly.tools.set_credentials_file(username='MaxPoole', api_key='2ajqCLZjiLNDFxgyLtGn')

    def generate_heat_map(self, data, axis_labels):
        """
        Generates a heat map for the current data
        Currently only meant to support using a cosine similarity matrix
        :param data:
        :param axis_labels:
        :return:
        """
        axis_labels_abbreviated = [label[:14] for label in axis_labels]
        info = [go.Heatmap(z=data,
                           x=axis_labels_abbreviated,
                           y=axis_labels_abbreviated,
                           colorscale = 'Hot',
                           )]

        layout = go.Layout(title='Cosine Similarity Between Documents',
                           xaxis = dict(ticks=' '),
                           yaxis = dict(ticks=' '),
                           plot_bgcolor= '#444',
                           paper_bgcolor= '#eee'
                           )
        fig = go.Figure(data=info, layout=layout)
        plotly.offline.plot(fig,filename='malware_heatmap.html')

    def k_means_clustering(self, factor_matrix):
        adjusted_weights = []
        for entry in factor_matrix:
            adjusted_weight = []
            adjusted_weight.append(sum(entry[:len(entry)//2])/(len(entry)/2))
            adjusted_weight.append(sum(entry[len(entry)//2:])/(len(entry)/2))
            adjusted_weights.append(adjusted_weight)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(factor_matrix)
        thing = [entry[0] for entry in adjusted_weights]
        data = [plotly.graph_objs.Scatter(x=[entry[0] for entry in adjusted_weights],
                                          y=[entry[1] for entry in adjusted_weights],
                                          mode='markers',
                                          marker=dict(color=kmeans.labels_)
                                          )
                ]
        fig = go.Figure(data=data)
        plotly.offline.plot(fig, filename='kmeans_cluster.html')



    def show(self):
        plt.show()


