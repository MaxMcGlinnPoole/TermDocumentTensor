import plotly
import plotly.graph_objs as go
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import time
import scipy.cluster.hierarchy
import scipy.spatial.distance
from collections import Counter


def flag_function_visualization(cmts):
    global flag
    flag = cmts


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
        if flag == 1:
            print("Generating Heatmap ")
        axis_labels_abbreviated = [label[:14] for label in axis_labels]
        info = [go.Heatmap(z=data,
                           x=axis_labels_abbreviated,
                           y=axis_labels_abbreviated,
                           colorscale='Hot',
                           )]

        layout = go.Layout(title='Cosine Similarity Between Documents',
                           xaxis=dict(ticks=' '),
                           yaxis=dict(ticks=' '),
                           plot_bgcolor='#444',
                           paper_bgcolor='#eee'
                           )
        fig = go.Figure(data=info, layout=layout)
        plotly.offline.plot(fig, filename='malware_heatmap.html')

    def k_means_clustering(self, factor_matrix, file_names=[], clusters=2):
        svd = TruncatedSVD(n_components=clusters, n_iter=7, random_state=42)
        reduced = svd.fit_transform(factor_matrix)
        kmeans = KMeans(n_clusters=clusters, random_state=0).fit(factor_matrix)
        labels_predicted = kmeans.labels_
        res = Counter(kmeans.labels_)
        val = list(res.values())
        president_data = 0
        shakespeare_data = 0
        president_data_correct = 0
        shakespeare_data_correct = 0

        for i in range(0, len(file_names)):
            filename = file_names[i]
            str = filename[1:4]
            if TensorVisualization.RepresentsInt(str):
                president_data = president_data + 1
                if labels_predicted[i] == 0:
                    president_data_correct = president_data_correct + 1
            else:
                shakespeare_data = shakespeare_data + 1
                if labels_predicted[i] == 1:
                    shakespeare_data_correct = shakespeare_data_correct + 1

        print("Number of president text files are", president_data)
        print("Number of president text files predicted correctly is", president_data)
        print("Accuracy : ", (president_data_correct / president_data) * 100)

        print("Number of shakespeare text files are", shakespeare_data)
        print("Number of shakespeare text files predicted correctly is", shakespeare_data_correct)
        print("Accuracy : ", (shakespeare_data_correct / shakespeare_data) * 100)

        data = [plotly.graph_objs.Scatter(x=[entry[0] for entry in reduced],
                                          y=[entry[1] for entry in reduced],
                                          mode='markers',
                                          marker=dict(color=kmeans.labels_),
                                          text=file_names
                                          )
                ]
        fig = go.Figure(data=data)
        plotly.offline.plot(fig, filename='kmeans_cluster.html')

    def RepresentsInt(s):
        try:
            int(s)
            return True
        except ValueError:
            return False

