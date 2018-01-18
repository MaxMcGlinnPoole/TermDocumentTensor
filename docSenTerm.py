import os
from tensorly.decomposition import parafac
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import _pickle as pickle
def main():
    tds = None
    cutoffs = []
    doc_content = []
    pos = 0
    max_matrix_height = 0
    times = 0
    file = open('php_data1503-100-5.pkl', 'rb')
    tds = pickle.load(file)
    print(tds)

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
        if times >= 5:
            break
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
            tds = np.dstack((tds, matrix))

    test = parafac(tds, rank=2)
    file = "php_data" + '-'.join(map(str, tds.shape)) + ".pkl"
    pickle.dump(tds, open(file, "wb"))
    print(test)
main()