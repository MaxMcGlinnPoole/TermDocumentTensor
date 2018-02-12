import plotly
import plotly.graph_objs as go
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import numpy as np



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
        print("clusters is " + str(clusters))
        svd = TruncatedSVD(n_components=clusters, n_iter=7, random_state=42)
        reduced = svd.fit_transform(factor_matrix)
        kmeans = KMeans(n_clusters=clusters, random_state=0).fit(factor_matrix)
        labels = kmeans.labels_
        if clusters == 2:
            data = [go.Scatter(x=[entry[0] for entry in reduced],
                                              y=[entry[1] for entry in reduced],
                                              mode='markers',
                                              marker=dict(
                                                  color=labels.astype(np.float),
                                                  line=dict(color='black', width=1)),
                                              text=file_names
                                              )
                    ]
        elif clusters == 3:
            data = [go.Scatter3d(x=[entry[0] for entry in reduced],
                                 y=[entry[1] for entry in reduced],
                                 z=[entry[2] for entry in reduced],
                                 showlegend=False,
                                 mode='markers',
                                 marker=dict(
                                     color=labels.astype(np.float),
                                     line=dict(color='black', width=1),

                                 ),
                                text=file_names)]

        fig = go.Figure(data=data)
        plotly.offline.plot(fig, filename='kmeans_cluster.html')
