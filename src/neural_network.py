import contextlib
import json
import os
from abc import abstractmethod

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.models import Sequential, load_model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

from src.helper import log, get_progress, get_time
from src.reduce_dimensions import LDA


class NeuralNetwork:
    """Base class for neural networks. Used to create a network and train it"

        Attributes:
        learning_rate           Learning rate used during training.
        epochs                  Number of epochs used during training.
        batch_size              Batch size used during training.
        image_iterator_bs       Batch size used for the image generator.
        directory               The directory where the input images (classes) live.
        train_generator         The generator used to create images for training.
        validation_generator    The generator used to create images for validation.
        cnn                     The actual network (Sequential).
        img_width               Width to which images are resized.
        img_height              Height to which images are resized.
        img_depth               Depth to which images are resized.
        classcount              The number of classes.
        history                 History where accuracies are saved during training.
        dimension_reducer       DimensionReducer used to reduce dimensions if needed for plotting.
        network  The locale where these birds congregate to reproduce.
    """
    learning_rate, epochs, batch_size, image_iterator_bs = 0.0001, 35, 64, 32
    directory = ''
    train_generator, validation_generator, cnn = None, None, None
    img_width, img_height, img_depth, classcount = 0, 0, 0, 0
    history, dimension_reducer, network = None, None, None

    def __init__(self, di, w=0, h=0, d=0, c=0):
        self.img_width = w
        self.img_height = h
        self.img_depth = d
        self.classcount = c
        self.directory = di
        self.build_network()
        self.validation_generator = ImageDataGenerator(validation_split=.2, rescale=1 / 255)
        self.train_generator = ImageDataGenerator(
            validation_split=.2,
            rescale=1 / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

    @abstractmethod
    def build_network(self):
        """
        Should be implemented for specific implementations of this baseclass
        """
        pass

    def create_dir_iterator(self, subset, image_generator, batch_size=None):
        """
        Create a directory iterator used for training
        :param subset: Training or validation set
        :param image_generator: Image generator to use
        :param batch_size: Generator batch size to use
        :return: Result of flow_from_directory
        """
        batch_size = self.image_iterator_bs if batch_size is None else batch_size
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            return image_generator.flow_from_directory(
                self.directory,
                color_mode='rgb' if self.img_depth == 3 else 'grayscale',
                subset=subset,
                batch_size=batch_size,
                target_size=(self.img_width, self.img_height),
                follow_links=True,
                class_mode='categorical'
            )

    def save_training_details(self, lr, e, b, igb):
        """
        Save parameters needed for training
        :param lr: Learning rate
        :param e:  Number of epochs
        :param b:  Batch size
        :param igb: Image generator batch size
        """
        self.learning_rate = lr
        self.epochs = e
        self.batch_size = b
        self.image_iterator_bs = igb

    def train(self, lr, e, b, igb, es_patience, rlr_patience):
        """
        Train the Sequential
        :param lr: Learning rate
        :param e: Number of epochs
        :param b: Batch size
        :param igb: Image generator batch size
        :param es_patience: Patience before terminating training
        :param rlr_patience:  Patience before lowering learning rate
        """
        self.save_training_details(lr, e, b, igb)

        train_generator = self.create_dir_iterator('training', self.train_generator)
        validation_generator = self.create_dir_iterator('validation', self.validation_generator)

        train_size = len(train_generator.filenames)
        validation_size = len(validation_generator.filenames)

        loss = 'categorical_crossentropy'
        opt = SGD(lr=self.learning_rate)
        # opt = Adam(lr=self.learning_rate, decay=self.learning_rate / self.epochs)

        self.network.compile(loss=loss,
                             optimizer=opt, metrics=['accuracy'])

        self.history = self.network.fit_generator(
            train_generator,
            steps_per_epoch=train_size // self.batch_size,
            epochs=self.epochs,
            validation_data=validation_generator,
            validation_steps=validation_size // self.batch_size,
            callbacks=[
                EarlyStopping(patience=es_patience, restore_best_weights=True),
                ReduceLROnPlateau(patience=rlr_patience)
            ],
        )

    def save_dimension_reducer(self, dimensions):
        """
        Create LDA for this network, later used to make predictions while plotting
        :param dimensions: Number of dimensions to reduce to, should be 2 or 3.
        """
        predictions, labels = [], []
        train_generator = self.create_dir_iterator('training', self.validation_generator, 32)
        total = train_generator.n // 32
        element = next(train_generator, None)
        i = 1
        while element is not None and i < total:
            log(f'performing LDA {get_progress(i / total)}', end='\r')
            labels.extend(element[1])
            predictions.extend(self.network.predict(element[0]))
            try:
                element = next(train_generator, None)
            except Exception:
                continue
            i += 1
        print('')
        self.dimension_reducer = LDA.apply(predictions, labels, dimensions)

    @staticmethod
    def load(filename, directory):
        """
        Instead of creating a new network, load it from a .h5 file
        :param filename: Filename of the .h5 file
        :param directory: Directory where the images used for trainnig the network live
        :return: Object of type NeuralNetwork
        """
        self = NeuralNetwork(directory)
        self.network = load_model(filename)
        shape = self.network.input_shape
        if not len(shape) == 4:
            raise Exception('Invalid network shape!')
        self.img_width, self.img_height, self.img_depth = shape[1], shape[2], shape[3]
        self.classcount = len(list(os.walk(directory))) - 1

        return self

    def save(self, filename):
        self.network.save(filename)

    def add_relu(self):
        self.network.add(Activation('relu'))

    def add_batch_normalization(self):
        self.network.add(BatchNormalization())

    def add_pooling(self, size):
        self.network.add(MaxPooling2D(pool_size=(size, size), strides=2))

    def add_dropout(self, amount):
        self.network.add(Dropout(amount))

    def add_flatten(self):
        self.network.add(Flatten())

    def add_fully_connected(self, outputsize):
        self.network.add(Dense(outputsize))

    def add_conv(self, nr_filters):
        self.network.add(Conv2D(
            filters=nr_filters,
            kernel_size=(3, 3),
            padding='same',
        ))


class SimpleCnn(NeuralNetwork):
    """Simple custom CNN, used by default, based on CopiedCnn"""

    def build_network(self):
        self.network = Sequential()

        self.network.add(Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
            input_shape=(self.img_width, self.img_height, self.img_depth),
            data_format='channels_last'))
        self.add_relu()
        self.add_pooling(3)
        self.add_dropout(.25)

        self.add_conv(64)
        self.add_relu()
        self.add_pooling(3)
        self.add_dropout(.25)

        self.add_flatten()

        self.add_fully_connected(64)

        self.add_relu()

        self.add_dropout(0.25)

        self.add_fully_connected(self.classcount)

        self.network.add(Activation("sigmoid"))
        return self.network


class CopiedCnn(NeuralNetwork):
    """Simple CNN, copied from
    https://towardsdatascience.com/classify-butterfly-images-with-deep-learning-in-keras-b3101fe0f98"""

    def build_network(self):
        cnn = Sequential()
        cnn.add(Conv2D(filters=32,
                       kernel_size=(2, 2),
                       strides=(1, 1),
                       padding='same',
                       input_shape=(self.img_width, self.img_height, self.img_depth),
                       data_format='channels_last'))
        cnn.add(Activation('relu'))
        cnn.add(MaxPooling2D(pool_size=(2, 2),
                             strides=2))
        cnn.add(Conv2D(filters=64,
                       kernel_size=(2, 2),
                       strides=(1, 1),
                       padding='valid'))
        cnn.add(Activation('relu'))
        cnn.add(MaxPooling2D(pool_size=(2, 2),
                             strides=2))
        cnn.add(Flatten())
        cnn.add(Dense(64))
        cnn.add(Activation('relu'))
        cnn.add(Dropout(0.25))
        cnn.add(Dense(self.classcount))
        cnn.add(Activation('sigmoid'))
        self.network = cnn
        return self.network


def parse_input(args):
    """
    Parse commandline arguments and create network
    :param args: Commandline arguments
    """
    image_width = args.image_width
    image_height = args.image_height
    image_depth = args.image_depth
    directory = args.input
    classcount = len(list(os.walk(directory))) - 1

    learning_rate = args.learning_rate
    epochs = args.epochs
    batch_size = args.batch_size
    generator_batch_size = args.image_generator_batch_size
    early_stopping_patience = args.early_stopping_patience
    reduce_lr_patience = args.reduce_lr_patience

    log(f'Creating neural network '
        f'with {classcount} clusters '
        f'for images in directory {directory} '
        f'with sizes {(image_width, image_height, image_depth)}')
    network = SimpleCnn(directory, image_width, image_height, image_depth, classcount)
    # network = NeuralNetwork.load()

    log(f'Training neural network '
        f'with learning rate {learning_rate}, '
        f'{epochs} epochs, '
        f'batch size of {batch_size}, '
        f'generator batch size of {generator_batch_size}, '
        f'patience before early stopping of {early_stopping_patience}, '
        f'and patience before reducing lr of {reduce_lr_patience}'
        )
    network.train(learning_rate, epochs, batch_size, generator_batch_size, early_stopping_patience, reduce_lr_patience)
    if not os.path.exists('../networks'):
        os.makedirs('../networks')
    filename = get_time()
    log(f'Saved network to networks{os.sep}{filename}.h5')
    with open(f'networks{os.sep}{filename}.json', 'w') as f:
        f.write(json.dumps(str(network.history.history)))
    network.save(f'networks{os.sep}{filename}.h5')
