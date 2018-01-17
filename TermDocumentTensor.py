from tensorly.tenalg import khatri_rao
from scipy import spatial
from collections import deque
import os
from tensorly.decomposition import parafac
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import _pickle as pickle

class TermDocumentTensor():
    def __init__(self, directory, type="binary"):
        self.vocab = []
        self.tensor = []
        self.corpus_names = []
        self.directory = directory
        self.type = type
        self.rank_approximation = None
        self.factor_matrices = []
        # These are the output of our tensor decomposition.
        self.factors = []

    def create_factor_matrices(self):
        tdm_1 = np.matmul(self.factors[0], np.transpose(khatri_rao([self.factors[2], self.factors[1]])))
        tdm_2 = np.matmul(self.factors[1], np.transpose(khatri_rao([self.factors[2], self.factors[0]])))
        tdm_3 = np.matmul(self.factors[2], np.transpose(khatri_rao([self.factors[1], self.factors[0]])))
        self.factors = [tdm_1, tdm_2, tdm_3]
        return self.factors

    def generate_cosine_similarity_matrix(self, matrix):
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
        for matrix in self.tensor:
            print(self.vocab)
            for i in range(len(matrix)):
                print(self.corpus_names[i], matrix[i])

    def create_term_document_tensor(self, **kwargs):
        if self.type == "binary":
            return self.create_binary_term_document_tensor(**kwargs)
        else:
            return self.create_text_corpus(**kwargs)

    def create_binary_term_document_tensor(self, **kwargs):
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
        return self.tensor

    def create_term_document_tensor_text(self):
        self.tensor = None
        cutoffs = []
        doc_content = []
        pos = 0
        max_matrix_height = 0
        times = 0
        # file = open('php_data.pkl', 'r')
        # self.tensor = pickle.load(file)
        print(self.tensor)

        for file_name in os.listdir(self.directory):
            cutoffs.append(pos)
            times += 1
            with open(self.directory + "/" + file_name, "r", encoding='latin-1') as file:
                for line in file:
                    if len(line) > 2:
                        pos += 1
                        doc_content.append(line)
                if max_matrix_height < pos - cutoffs[-1]:
                    max_matrix_height = pos - cutoffs[-1]
        cutoffs.append(pos)
        vectorizer = TfidfVectorizer(use_idf=False, analyzer="word")

        x1 = vectorizer.fit_transform(doc_content)
        matrix_length = len(vectorizer.get_feature_names())
        svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
        for i in range(len(cutoffs) - 1):
            temp = x1[cutoffs[i]:cutoffs[i + 1], :]

            temp = temp.todense()
            matrix = np.zeros((max_matrix_height, matrix_length))
            matrix[:temp.shape[0], :temp.shape[1]] = temp
            # reduce dimensionality of matrix slices to reduce memory overhead
            matrix = svd.fit_transform(matrix)
            if self.tensor is None:
                self.tensor = matrix
            else:
                self.tensor = np.append(self.tensor, matrix, axis=0)

        test = parafac(self.tensor, rank=2)
        pickle.dump(self.tensor, open("php_data.pkl", "w"))
        print(test)
        return self.tensor

    def parafac_decomposition(self):
        self.factors = parafac(np.array(self.tensor), rank=self.get_estimated_rank())
        return self.factors

