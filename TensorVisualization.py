import plotly
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD



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

    def k_means_clustering(self, factor_matrix, file_names=[], clusters=2):
        svd = TruncatedSVD(n_components=clusters, n_iter=7, random_state=42)
        reduced = svd.fit_transform(factor_matrix)
        kmeans = KMeans(n_clusters=clusters, random_state=0).fit(factor_matrix)
        data = [plotly.graph_objs.Scatter(x=[entry[0] for entry in reduced],
                                          y=[entry[1] for entry in reduced],
                                          mode='markers',
                                          marker=dict(color=kmeans.labels_),
                                          text=file_names
                                          )
                ]
        fig = go.Figure(data=data)
        plotly.offline.plot(fig, filename='kmeans_cluster.html')
