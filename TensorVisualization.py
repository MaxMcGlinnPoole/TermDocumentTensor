import plotly
from plotly.graph_objs import *
import matplotlib.pyplot as plt


class TensorVisualization():
    def __init__(self):
        plotly.tools.set_credentials_file(username='MaxPoole', api_key='2ajqCLZjiLNDFxgyLtGn')

    def generate_heat_map(self, data):
        fig, ax1 = plt.subplots(1, 1)
        ax1.imshow(data, cmap='hot')

    def show(self):
        plt.show()


