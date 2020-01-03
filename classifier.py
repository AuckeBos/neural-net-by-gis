import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0, 1, 2, 3
import argparse
from src.get_google_images import parse_input as get_google_images
from src.crawl_terms import parse_input as crawl_terms
from src.neural_network import parse_input as create_network
from src.plotter import parse_input as create_plots


def _main():
    """
    Parse command line arguments and run selected command
    """
    parser = argparse.ArgumentParser(prog='classifier',
                                     description='Building a convolutional neural network using Google Image Search',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter
                                     )
    subparsers = parser.add_subparsers(dest='command', title='command', help='This library consists of the following '
                                                                             'functions:')
    subparsers.required = True

    get_image_parser = subparsers.add_parser('get_images', help='Retrieve images from Google Image Search',
                                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    get_image_parser.add_argument('-c', '--count',
                                  type=int,
                                  help='Amount of images to download per query',
                                  default=1000)
    get_image_parser.add_argument('input',
                                  help='Input file containing one query per line')
    get_image_parser.add_argument('output',
                                  help='Directory where images will be saved, each in their own sub directory')
    get_image_parser.set_defaults(func=get_google_images)

    crawl_terms_parser = subparsers.add_parser('crawl_terms', help='Crawl a list of queries from a table of a webpage; '
                                                                   'tested on wikipedia articles',
                                               formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    crawl_terms_parser.add_argument('url',
                                    help='Url to crawl terms from')
    crawl_terms_parser.add_argument('-o', '--output',
                                    help='File to save terms to',
                                    default='terms.txt')
    crawl_terms_parser.set_defaults(func=crawl_terms)

    create_network_parser = subparsers.add_parser('create_network', help='Create and train a neural network, and '
                                                                         'save its result to a file for later usage',
                                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    create_network_parser.add_argument('-w', '--image-width',
                                       type=int,
                                       help='Target width of the images; all images will be reshaped to this width',
                                       default=64)
    create_network_parser.add_argument('-H', '--image-height',
                                       type=int,
                                       help='Target height of the images; all images will be reshaped to this height',
                                       default=64)
    create_network_parser.add_argument('-d', '--image-depth',
                                       type=int,
                                       help='Target depth of the images; 3 for rgb, 1 for grayscale',
                                       default=3)
    create_network_parser.add_argument('input',
                                       help='Directory holding the images;'
                                            ' each class should have its own subdirectory')
    create_network_parser.add_argument('-l', '--learning-rate',
                                       type=float,
                                       help='Learningrate used for optimizing the neural network; '
                                            'note that the network applies ReduceOnPlateau',
                                       default=1e-4)
    create_network_parser.add_argument('-e', '--epochs',
                                       type=int,
                                       help='Number of epochs; note that the network will apply EarlyStopping',
                                       default=100)
    create_network_parser.add_argument('-b', '--batch-size',
                                       type=int,
                                       help='Batch size',
                                       default=1)
    create_network_parser.add_argument('-B', '--image-generator-batch-size',
                                       type=int,
                                       help='Number of elements in each generated batch of the generator',
                                       default=1)
    create_network_parser.add_argument('-p', '--early-stopping-patience',
                                       type=int,
                                       help='Patience for early stopping mechanism if accuracy doesn\'t seem to'
                                            'increase anymore',
                                       default=10)
    create_network_parser.add_argument('-P', '--reduce-lr-patience',
                                       type=int,
                                       help='Patience for reducing lr if accuracy doesn\'t seem to increase anymore',
                                       default=5)

    create_network_parser.set_defaults(func=create_network)

    create_plots_parser = subparsers.add_parser('create_plots', help='Load a network (possibly created by '
                                                                     'create_network), and show an accuracy plot '
                                                                     'and a scatter plot of clustering outputs of '
                                                                     'some random samples',
                                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    create_plots_parser.add_argument('network',
                                     help='.h5 file of the network to plot; note that the network assumes'
                                          'that a .json file exists with the same name in the same directory.'
                                          'It should contain the training history, which is used for creating'
                                          'the accuracy plot')
    create_plots_parser.add_argument('directory',
                                     help='Directory holding the images from which the network was created; used to'
                                          'sample random images for the scatter plot')
    create_plots_parser.add_argument('-C', '--nr-of-clusters-to-plot',
                                     type=int,
                                     help='Number of elements to include in scatter plot',
                                     default=100)
    create_plots_parser.add_argument('-D', '--cluster-plot-dimension',
                                     type=int,
                                     help='Number of dimensions for cluster plot, should be 2 or 3.',
                                     default=2)
    create_plots_parser.set_defaults(func=create_plots)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    _main()
