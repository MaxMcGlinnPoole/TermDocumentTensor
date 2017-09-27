import plotly
from plotly.graph_objs import *
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


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
        # This makes the labels appear correctly, I do not know why
        axis_labels_abbreviated = [""]
        axis_labels_abbreviated.extend([label[:5] for label in axis_labels])
        fig, ax1 = plt.subplots(1, 1)
        labels = ax1.get_xticks().tolist()
        plt.xticks(labels, axis_labels_abbreviated, rotation='vertical')
        plt.yticks(labels, axis_labels_abbreviated)

        ax1.yaxis.set_major_locator(MaxNLocator(len(axis_labels)))
        ax1.xaxis.set_major_locator(MaxNLocator(len(axis_labels)))
        ax1.imshow(data, cmap='hot')

    def show(self):
        plt.show()


