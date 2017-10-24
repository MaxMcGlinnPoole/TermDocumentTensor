import csv
import os
from tensorly.tenalg import khatri_rao
from tensorly.decomposition import parafac
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy import spatial
from collections import deque
import re
import TensorVisualization

import CSVConverter


class TermDocumentTensor():
    def __init__(self, directory, type="binary"):
        self.vocab = []
        self.tdt = []
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
                sim.append(spatial.distance.cosine(entry, other_entry)*-1 + 1)
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

        I = len(self.tdt[0])
        J = len(self.tdt[0][0])
        K = len(self.tdt)

        if I == 1 or J == 1 or K == 1:
            return 1
        elif I == J == K == 2:
            return 2
        elif I == J == 3 and K == 2:
            return 3
        elif I == 5 and J == K == 3:
            return 5
        elif I >= 2*J and K == 2:
            return 2*J
        elif 2*J > I > J and K ==2:
            return I
        elif I == J and K == 2:
            return I
        elif I >= J*K:
            return J*K
        elif J*K - J < I < J*K:
            return I
        elif I == J*K - I:
            return I
        else:
            print(I, J, K, "did not have an exact estimation")
            return min(I*J, I*K, J*K)

    def print_formatted_term_document_tensor(self):
        for matrix in self.tdt:
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
        self.vocab = ["vocab"]

        self.vocab.extend(vectorizer.get_feature_names())
        tdm = []
        for i in range(len(doc_names)):
            row = x1[i]
            tdm.append(row)
        tdm_first_occurences = []
        self.corpus_names = doc_names
        # tdm_first_occurences[0] = tdm[0]
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

        tdt = [tdm, tdm_first_occurences]
        self.tdt = tdt
        return self.tdt
        

    def create_term_document_tensor_text(self):
        mydoclist = []
        #tdm = textmining.TermDocumentMatrix()
        files = []
        first_occurences_corpus = {}
        text_names = []
        number_files = 0
        for file in os.listdir(self.directory):
            number_files += 1
            first_occurences = {}
            words = 0
            with open(self.directory + "/" + file, "r") as shake:
                files.append(file)
                lines_100 = ""
                while True:
                    my_line = shake.readline()
                    if not my_line:
                        break
                    re.sub(r'\W+', '', my_line)
                    for word in my_line.split():
                        words += 1
                        if word not in first_occurences:
                            first_occurences[word] = words
                    lines_100 += my_line
            first_occurences_corpus[file] = first_occurences
            tdm.add_doc(lines_100)
            mydoclist.append(file)
            text_names.append(file)
        tdm = list(tdm.rows(cutoff=1))
        tdt = [0, 0]
        tdm_first_occurences = []
        # tdm_first_occurences[0] = tdm[0]
        # Create a first occurences matrix that corresponds with the tdm
        for j in range(len(text_names)):
            item = text_names[j]
            this_tdm = []
            for i in range(0, len(tdm[0])):
                word = tdm[0][i]
                try:
                    this_tdm.append(first_occurences_corpus[item][word])
                except:
                    this_tdm.append(0)
            # print(this_tdm)
            tdm_first_occurences.append(this_tdm)
        self.vocab = tdm.pop(0)
        self.corpus_names = mydoclist
        tdt[0] = tdm
        tdt[1] = tdm_first_occurences
        tdt = np.asanyarray(tdt)
        self.tdt = tdt
        return tdt

    def parafac_decomposition(self):
        self.factors = parafac(np.array(self.tdt), rank=self.get_estimated_rank())
        test = np.array(self.tdt)
        print(test)
        return self.factors


def main():
    tdt = TermDocumentTensor("zeus_binaries", type="binary")
    tdt.create_binary_term_document_tensor(ngrams=1)
    print(tdt.get_estimated_rank())
    factors = tdt.parafac_decomposition()
    factor_matrices = tdt.create_factor_matrices()
    cos_sim = tdt.generate_cosine_similarity_matrix(factor_matrices[1])
    visualize = TensorVisualization.TensorVisualization()
    visualize.generate_heat_map(cos_sim, tdt.corpus_names)
    visualize.show()

main()