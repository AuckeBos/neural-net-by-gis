import contextlib
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator

from src.helper import log
from src.neural_network import NeuralNetwork


def _plot_accuracy(neural_network: NeuralNetwork, history, filename):
    """
    Create accuracy plot based on history of training of NeuralNetwork
    :param neural_network: NeuralNetwork to create plot for
    :param history: History of training the NeuralNetwork
    :param filename: Filename to save the plot to
    """
    plt.style.use('ggplot')
    plt.figure()
    N = len(history['val_loss'])
    plt.plot(np.arange(0, N), history['accuracy'], label='train_acc')
    plt.plot(np.arange(0, N), history['val_accuracy'], label='val_acc')
    plt.title(f'Accuracy plot for network {filename}\n'
              f' img size = {(neural_network.img_width, neural_network.img_height, neural_network.img_depth)}')
    plt.xlabel('Epoch #')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    if not os.path.exists('../accuracy_plots'):
        os.makedirs('../accuracy_plots')
    filename = f'accuracy_plots{os.sep}{filename}.png'
    plt.savefig(filename)
    log(f'Saved accuracy plot to {filename}')
    Image.open(filename).show()


def _get_labelname(class_indices, label_vector):
    """
    Convert one-hot encoding to lablename string
    :param class_indices: Class indexes as created by the ImageGenerator
    :param label_vector: Vector of the label to convert
    :return: Class as string
    """
    return class_indices[[x for x in range(len(label_vector)) if label_vector[x] == 1][0]]


def _get_predictions(neural_network: NeuralNetwork, nr_of_elements, nr_of_dimensions):
    """
    Precict ``nr_of_elements`` random elements by ``neural_network``
    :param neural_network: The NeuralNetwork to predict with
    :param nr_of_elements: Amount of elements to predict
    :param nr_of_dimensions: Number of dimensions to reduce preditions to
    :return:
    """
    should_reduce = nr_of_dimensions < neural_network.classcount
    if should_reduce:
        neural_network.save_dimension_reducer(nr_of_dimensions)
    data = {
        "x": [],
        "y": [],
        "true_label": [],
        "filename": [],
    }
    if nr_of_dimensions == 3:
        data["z"] = []
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        generator = ImageDataGenerator(rescale=1 / 255).flow_from_directory(
            neural_network.directory,
            color_mode="rgb",
            batch_size=32,
            shuffle=True,
            target_size=(neural_network.img_width, neural_network.img_height),
            follow_links=True,
            class_mode='categorical',
        )

    class_indices = {v: k for k, v in generator.class_indices.items()}
    for batch_index in range(nr_of_elements // 32):
        elements = generator.next()
        true_labels = [_get_labelname(class_indices, label) for label in elements[1]]
        predicted_labels = neural_network.network.predict(elements[0])
        dimensions_reduced = neural_network.dimension_reducer.transform(predicted_labels) \
            if should_reduce else predicted_labels

        data["x"].extend(dimensions_reduced[:, 0])
        data["y"].extend(dimensions_reduced[:, 1])
        if nr_of_dimensions == 3:
            data["z"].extend(dimensions_reduced[:, 2])
        data["true_label"].extend(true_labels)
        filenames = [generator.filenames[generator.index_array[file_index + 32 * batch_index]] for file_index in
                     range(32)]
        data["filename"].extend(filenames)
    return data


def _plot_clusters(neural_network: NeuralNetwork, nr_of_elements, nr_of_dimensions, filename):
    """
    Create scatter plot for ``nr_of_elements`` random elements
    :param neural_network: The NeuralNetwork to create the plot for
    :param nr_of_elements:  The number of elements to plot
    :param nr_of_dimensions:  The number of dimensions to plot in, should be 2 or 3
    :param filename: The filename used for the title of the plot
    """
    assert (1 < nr_of_dimensions < 4), "Can only plot in 2 or 3 dimensions!"
    data = _get_predictions(neural_network, nr_of_elements, nr_of_dimensions)
    df = pd.DataFrame(data)
    if nr_of_dimensions == 3:
        fig = px.scatter_3d(df, x="x", y="y", z="z",
                            title=f'Cluster plot for network {filename} with images of size '
                                  f'{(neural_network.img_width, neural_network.img_height, neural_network.img_depth)}',
                            color='true_label', hover_data=["filename"])
    else:
        fig = px.scatter(df, x="x", y="y",
                         title=f'Cluster plot for network {filename} with images of size '
                               f'{(neural_network.img_width, neural_network.img_height, neural_network.img_depth)}',
                         color='true_label', hover_data=["filename"])
    log('Showing cluster plot in browser')
    fig.show()
    # color_discrete_map={c:c for c in df["true_label"].unique()}


def parse_input(args):
    """
    Parse commandline arguments and create plots
    :param args: Commandline arguments
    """
    filename = args.network.replace('.h5', '')
    directory = args.directory
    nr_of_clusters_to_plot = args.nr_of_clusters_to_plot
    cluster_plot_dimensions = args.cluster_plot_dimension

    network = NeuralNetwork.load(f'{filename}.h5', directory)
    history = None
    if (os.path.exists(f'{filename}.json')):
        with open(f'{filename}.json') as f:
            history = json.loads(f.read().strip('"').replace("'", '"'))
    filename = filename.split(os.sep)[-1]
    if history is not None:
        _plot_accuracy(network, history, filename)
    _plot_clusters(network, nr_of_clusters_to_plot, cluster_plot_dimensions, filename)
