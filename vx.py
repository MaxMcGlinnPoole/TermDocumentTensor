import TermDocumentTensor
import TensorVisualization
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Optional arguments
    parser.add_argument("-d", "--directory", dest="directory",
                        help="Specify a directory where the samples are stored", required=True)

    parser.add_argument('-heatmap', dest="heatmap", help="Generates a HeatMap of the cosine similarity matrix",
                        action="store_true",
                        default=False)
    parser.add_argument("-kmeans", dest="kmeans", help="Tells the program to perform KMeans clustering",
                        action="store_true",
                        default=False)
    parser.add_argument("-comp", dest="components", type=int, help="Number of components for the Kmeans clustering",
                        default=2)
    parser.add_argument("-ngrams", "--ngrams", dest="ngrams", type=int,
                        help="Number of n-grams that will be used in the tensor creation", required=True)
    # Mutually exclusive arguments, in groups.
    # For each group, the first option is true by default,and the rest are false
    ft_group = parser.add_mutually_exclusive_group()
    ft_group.add_argument("-b", "--binary", dest="binary", help="Analyze binary files", action="store_true",
                          default=True)
    ft_group.add_argument("-t", "--text", dest="text", help="Analyze text files", action="store_true", default=False)

    decomp_group = parser.add_mutually_exclusive_group()
    decomp_group.add_argument("-parafac", dest="decom", help="Use a parafac decomposition", action="store_true",
                              default="parafac")

    parser.add_argument("-o", "--output", dest="output_option",
                        help="Specify whether to generate an output file", action="store_true")

    # Sample usage:  python3 vx.py -d myDirectory -v heatmap -b -parafac -o
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    file_type = "binary" if args.binary else "text"
    tdt = TermDocumentTensor.TermDocumentTensor(args.directory, type=file_type)
    tdt.create_binary_term_document_tensor(ngrams=args.ngrams)
    if args.decom == "parafac":
        factors = tdt.parafac_decomposition()
    cos_sim = None
    visualize = TensorVisualization.TensorVisualization()
    if args.heatmap:
        cos_sim = tdt.generate_cosine_similarity_matrix(factors[1])
        visualize.generate_heat_map(cos_sim, tdt.corpus_names)
    if args.kmeans:
        visualize.k_means_clustering(factors[1], tdt.corpus_names, clusters=args.components)


main()
