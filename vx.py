
import csv
from os import listdir
import re
import textmining
import TensorToolbox

def convertToCSV(term_document_matrix, files):
    # Converts a tdm to csv
    with open("test.csv", "w", newline='') as csv_file:
        writer = csv.writer(csv_file)
        for entry in term_document_matrix:
            num_list = map(str, entry)
            writer.writerow(num_list)


def main():
    mydoclist = []
    tdm = textmining.TermDocumentMatrix()
    files = []
    first_occurences_corpus = {}
    text_names = []
    for file in listdir("Folger"):
        first_occurences = {}
        words = 0
        with open("Folger/" + file, "r") as shake:
            files.append(file)
            lines_100 = ""
            for i in range(100):
                my_line = shake.readline()
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
    print(first_occurences_corpus)
    tdm = list(tdm.rows(cutoff=1))
    tdt = [0,0]
    tdm_first_occurences = [0]
    tdm_first_occurences[0] = tdm[0]
    #Create a first occurences matrix that corresponds with the tdm
    for j in range(len(text_names)):
        item = text_names[j]
        this_tdm = []
        for i in range(0, len(tdm[0])):
            word = tdm[0][i]
            try:
               this_tdm.append(first_occurences_corpus[item][word])
            except:
                this_tdm.append(0)
        #print(this_tdm)
        tdm_first_occurences.append(this_tdm)
    for row in tdm_first_occurences:
        print(row)
    tdt[0] = tdm
    tdt[1] = tdm_first_occurences
    print(tdt)
main()
