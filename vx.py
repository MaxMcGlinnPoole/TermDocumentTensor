import TermDocumentTensor
import TensorVisualization
import argparse
import time


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Optional arguments
    parser.add_argument("-d", "--directory", dest="directory",
                        help="Specify a directory where the samples are stored", required=True)
    parser.add_argument("-f", "--file", dest="file",
                        help="Specify a pickle file from which the tensor should be read")

    parser.add_argument('-heatmap', dest="heatmap", help="Generates a HeatMap of the cosine similarity matrix",
                        action="store_true",
                        default=False)
    parser.add_argument("-kmeans", dest="kmeans", help="Tells the program to perform KMeans clustering",
                        action="store_true",
                        default=False)
    parser.add_argument("-axis", dest="axis", help="The axis on which the decomposition should be modelled. "
                                                   "This number corresponds with the same entry factor matrix"
                                                   "from the decomposition. By default this is 1", default=1, type=int)
    parser.add_argument("-lines", dest="lines", help="The max number of lines that should be read from each file to "
                                                     "make the decomposition. By default this is 100",
                        default=100, type=int)
    parser.add_argument("-Comments", dest="Comments", help="y/n if you want comments ",

                        default='Yes', type=str)
    parser.add_argument("-comp", dest="components", type=int, help="Number of components for the Kmeans clustering",
                        default=2)
    parser.add_argument("-ngrams", "--ngrams", dest="ngrams", type=int,
                        help="Number of n-grams that will be used in the tensor creation", required=False, default=1)

    # Mutually exclusive arguments, in groups.
    # For each group, the first option is true by default,and the rest are false
    ft_group = parser.add_mutually_exclusive_group()
    ft_group.add_argument("-b", "--binary", dest="binary", help="Analyze binary files", action="store_true",
                          default=False)
    ft_group.add_argument("-t", "--text", dest="text", help="Analyze text files", action="store_true", default=False)

    decomp_group = parser.add_mutually_exclusive_group()
    decomp_group.add_argument("-parafac", dest="decom", help="Use a parafac decomposition", action="store_true",
                              default="parafac")

    parser.add_argument("-o", "--output", dest="output_option",
                        help="Specify whether to generate an output file", action="store_true")

    # Sample usage:  python3 vx.py -d myDirectory -v heatmap -b -parafac -o
    args = parser.parse_args()
    return args


def display_info_message(args):
    if args.binary and not args.ngrams:
        print("WARNING: Constructing a tensor from binary executables with no provided ngrams. Using ngram=1 as default")


def main():
    start_time = time.time()
    args = parse_arguments()
    flag = args.Comments

    if flag[0] == "y" or flag[0] == "Y":
        flag = 1
    else:
        flag = 0

    TermDocumentTensor.flag_function_tdm(flag)
    TensorVisualization.flag_function_visualization(flag)

    if flag == 1:
        print("About to run the term document tensor")

    file_type = "binary" if args.binary else "text"
    tdt = TermDocumentTensor.TermDocumentTensor(args.directory, type=file_type, file_name=args.file)
    tdt.create_term_document_tensor(ngrams=args.ngrams, lines=args.lines)

    if args.decom == "parafac":
        factors = tdt.parafac_decomposition()

    visualize = TensorVisualization.TensorVisualization()
    if args.heatmap:
        cos_sim = tdt.generate_cosine_similarity_matrix(factors[args.axis])
        visualize.generate_heat_map(cos_sim, tdt.corpus_names)
    if args.kmeans:
        visualize.k_means_clustering(factors[args.axis], tdt.corpus_names, clusters=args.components)
    if flag == 1:
        print("  %s seconds is the total time for program to execute" % format((time.time() - start_time), '.2f'))


main()
