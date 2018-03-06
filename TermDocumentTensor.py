from scipy import spatial
from collections import deque
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import _pickle as pickle
import tensorflow as tf
from ktensor import KruskalTensor
import time


def flag_function_tdm(cmts):
        global flag
        flag = cmts


class TermDocumentTensor():
    def __init__(self, directory, type="binary", file_name=None):
        if flag==1:
            print("Initilizing the Tensor")
        self.vocab = []
        self.tensor = []
        self.corpus_names = []
        self.directory = directory
        self.type = type
        self.rank_approximation = None
        self.factor_matrices = []
        # These are the output of our tensor decomposition.
        self.factors = []
        self.file_name = file_name

    def generate_cosine_similarity_matrix(self, matrix):
        if flag == 1:
            print("Generating a cosine similarity matrix")
        cosine_sim = []
        for entry in matrix:
            sim = []
            for other_entry in matrix:
                sim.append(spatial.distance.cosine(entry, other_entry) * -1 + 1)
            cosine_sim.append(sim)
        return cosine_sim

    def get_estimated_rank(self):
        """
        Getting the rank of a tensor is an NP hard problem
        Therefore we use an estimation based on the size of the dimensions of our tensor.
        These numbers are grabbed from Table 3.3 of Tammy Kolda's paper:
        http://www.sandia.gov/~tgkolda/pubs/pubfiles/TensorReview.pdf
        :return:
        """
        # At the moment the rank returned by this function is normally too high for either
        # my machine or the tensorly library to handle, therefore I have made it just return 1 for right now
        if flag == 1:
            print("Estimation of the rank of tensor ")
        I = len(self.tensor[0])
        J = len(self.tensor[0][0])
        K = len(self.tensor)

        if I == 1 or J == 1 or K == 1:
            return 1
        elif I == J == K == 2:
            return 2
        elif I == J == 3 and K == 2:
            return 3
        elif I == 5 and J == K == 3:
            return 5
        elif I >= 2 * J and K == 2:
            return 2 * J
        elif 2 * J > I > J and K == 2:
            return I
        elif I == J and K == 2:
            return I
        elif I >= J * K:
            return J * K
        elif J * K - J < I < J * K:
            return I
        elif I == J * K - I:
            return I
        else:
            print(I, J, K, "did not have an exact estimation")
            return min(I * J, I * K, J * K)

    def print_formatted_term_document_tensor(self):
        if flag == 1:
            print("Print the TDM")
        for matrix in self.tensor:
            print(self.vocab)
            for i in range(len(matrix)):
                print(self.corpus_names[i], matrix[i])

    def create_term_document_tensor(self, **kwargs):
        """
        Generic tensor creation function. Returns different tensor based on user input.
        :param kwargs:
        :return:
        """
        if flag == 1:
            print("Creating a TermDocumentTensor")
        if self.type == "binary":
            return self.create_binary_term_document_tensor(**kwargs)
        else:
            return self.create_term_document_tensor_text(**kwargs)

    def create_binary_term_document_tensor(self, **kwargs):
        start_time1 = time.time()
        if flag == 1:
            print("Binary TermDocumentTensor")
        doc_content = []
        first_occurences_corpus = {}
        ngrams = kwargs["ngrams"] if kwargs["ngrams"] is not None else 1
        print(ngrams)

        for file_name in os.listdir(self.directory):
            previous_bytes = deque()
            first_occurences = {}
            byte_count = 0
            with open(self.directory + "/" + file_name, "rb") as file:
                my_string = ""
                while True:
                    byte_count += 1
                    current_byte = file.read(1).hex()
                    if not current_byte:
                        break
                    if byte_count >= ngrams:
                        byte_gram = "".join(list(previous_bytes)) + current_byte
                        if byte_gram not in first_occurences:
                            first_occurences[byte_gram] = byte_count
                        if byte_count % ngrams == 0:
                            my_string += byte_gram + " "
                        if ngrams > 1:
                            previous_bytes.popleft()
                    if ngrams > 1:
                        previous_bytes.append(current_byte)
                first_occurences_corpus[file_name] = first_occurences
            doc_content.append(my_string)
        doc_names = os.listdir(self.directory)

        # Convert a collection of text documents to a matrix of token counts
        vectorizer = TfidfVectorizer(use_idf=False)
        # Learn the vocabulary dictionary and return term-document matrix.
        x1 = vectorizer.fit_transform(doc_content).toarray()
        del doc_content
        self.vocab = ["vocab"]

        self.vocab.extend(vectorizer.get_feature_names())
        tdm = []
        for i in range(len(doc_names)):
            row = x1[i]
            tdm.append(row)
        svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
        reduced_tdm = svd.fit_transform(tdm)
        tdm_first_occurences = []
        self.corpus_names = doc_names
        # Create a first occurences matrix that corresponds with the tdm
        for j in range(len(doc_names)):
            item = doc_names[j]
            this_tdm = []
            for i in range(0, len(tdm[0])):
                word = self.vocab[i]
                try:
                    this_tdm.append(first_occurences_corpus[item][word])
                except:
                    this_tdm.append(0)
            # print(this_tdm)
            tdm_first_occurences.append(this_tdm)
        reduced_tdm_first_occurences = svd.fit_transform(tdm_first_occurences)
        del tdm_first_occurences
        del tdm
        tdt = [reduced_tdm, reduced_tdm_first_occurences]
        self.tensor = tdt
        #tdm_sparse = scipy.sparse.csr_matrix(tdm)
        #tdm_first_occurences_sparse = scipy.sparse.csr_matrix(tdm_first_occurences)
        if flag == 1:
            print("  %s seconds for TDM Binary" % format((time.time() - start_time1),'.2f'))
        return self.tensor

    def create_term_document_tensor_text(self, **kwargs):
        """
        Creates term-sentence-document tensor out of files in directory
        Attempts to save this tensor to a pickle file
        
        :return: 3-D dense numpy array, self.tensor
        """
        start_time2 = time.time()
        if flag == 1:
            print("Document TermDocumentTensor")

        self.tensor = None
        vectorizer = TfidfVectorizer(use_idf=False, analyzer="word")
        document_cutoff_positions = []
        doc_content = []
        pos = 0
        max_matrix_height = 0
        max_sentences = kwargs["lines"]
        self.corpus_names = os.listdir(self.directory)

        # If given Pickle file, read it in
        if self.file_name is not None:
            file = open(self.file_name, 'rb')
            self.tensor = pickle.load(file)
            return self.tensor

        # Create one large term document matrix from all documents. Done to ensure same vocabulary.
        for file_name in self.corpus_names:
            document_cutoff_positions.append(pos)
            with open(self.directory + "/" + file_name, "r", errors="ignore") as file:
                for line in file:
                    if len(line) > 2:
                        pos += 1
                        doc_content.append(line)
                    if pos - document_cutoff_positions[-1] >= max_sentences:
                        break
                if max_matrix_height < pos - document_cutoff_positions[-1]:
                    max_matrix_height = pos - document_cutoff_positions[-1]

        document_cutoff_positions.append(pos)

        x1 = vectorizer.fit_transform(doc_content)
        matrix_length = len(vectorizer.get_feature_names())

        # Split large term document matrix, into term document tensor. Splits happen where one document ends.
        for i in range(len(document_cutoff_positions) - 1):
            temp = x1[document_cutoff_positions[i]:document_cutoff_positions[i + 1], :]
            temp = temp.todense()
            # Make all matrix slices the same size
            term_sentence_matrix = np.zeros((max_matrix_height, matrix_length))
            term_sentence_matrix[:temp.shape[0], :temp.shape[1]] = temp
            if self.tensor is None:
                self.tensor = term_sentence_matrix
            else:
                self.tensor = np.dstack((self.tensor, term_sentence_matrix))

        self.file_name = self.directory + ".pkl"
        if flag == 1:
            print("Finished tensor construction.")
        if flag == 1:
            print("Tensor shape:" + str(self.tensor.shape))
        try:
            pickle.dump(self.tensor, open(self.file_name, "wb"))
        except OverflowError:
            print("ERROR: Tensor cannot be saved to pickle file due to size larger than 4 GiB")
        if flag == 1:
            print("  %s seconds for TDM document" % format((time.time() - start_time2),'.2f'))
        return self.tensor

    def parafac_decomposition(self):
        """
        Computes a parafac decomposition of the tensor.
        This will return n rank 3 factor matrices. Where n represents the dimensionality of the tensor.
        :return:
        """
        start_time3 = time.time()
        if flag == 1:
            print("In decomposition of a TDM")
        decompose = KruskalTensor(self.tensor.shape, rank=3, regularize=1e-6, init='nvecs', X_data=self.tensor)
        self.factors = decompose.U
        with tf.Session() as sess:
            for i in range(len(self.factors)):
                sess.run(self.factors[i].initializer)
                self.factors[i] = self.factors[i].eval()
        if flag == 1:
            print("  %s seconds for decomposition of a tensor" % format((time.time() - start_time3), '.2f'))
        return self.factors

