import os
from tensorly.decomposition import parafac
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def main():
    tds = None
    cutoffs = []
    doc_content = []
    pos = 0
    max_matrix_height = 0
    times = 0
    for file_name in os.listdir("php"):
        cutoffs.append(pos)
        times += 1
        with open("php/" + file_name, "r", encoding='latin-1') as file:
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
    for i in range(len(cutoffs)-1):
        temp = x1[cutoffs[i]:cutoffs[i+1],:]

        temp = temp.todense()
        matrix = np.zeros((max_matrix_height, matrix_length))
        matrix[:temp.shape[0], :temp.shape[1]] = temp
        # reduce dimensionality of matrix slices to reduce memory overhead
        matrix = svd.fit_transform(matrix)
        if tds is None:
            tds = matrix
        else:
            tds = np.append(tds, matrix, axis=0)

    test = parafac(tds, rank=2)
    np.savetxt("php_data.txt", tds)
    print(test)
main()