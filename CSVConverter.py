import csv
import glob
import ntpath


def convert_term_document_tensor_to_csv(tdt, file_name):
    """
    Converts the term document matrix of a term document tensor to a csv file
    :param tdt: The term document tensor
    :param file_name: The name of the file that will be written to
    :return: None
    """
    print("converting the TDM to csv")
    if isinstance(tdt[0][0], list):
        tdt = tdt[0]
    with open(file_name + ".csv", "w", newline='') as csv_file:
        writer = csv.writer(csv_file)
        for entry in tdt:
            num_list = map(str, entry)
            writer.writerow(num_list)


def generate_term_list_csv(corpus_name, file_name, number_terms = 61, binary=True,**kwargs):
    """
    Creates a csv file that simply lists the terms in the documents in order
    :param corpus_name: Name of the directory that contains the files to be read
    :param number_terms: The number of terms to be read in a row.
                         A default value of 61 is given due to the limitations of the ensign tool.
    :param file_name: The name of the csv file that the term list will be written to
    :param number_terms: The number of terms to write to the csv file
    :param numerical_rep: Whether or not we should convert the hex bytes in the binary file into a integer.
                          May be useful for some representation methods
    :return: None
    """
    print("Generating the term,list of csv document")
    if binary:
        generate_binary_term_list(corpus_name, file_name, **kwargs)
    else:
        generate_text_term_list(corpus_name, file_name,  **kwargs)


def generate_binary_term_list(corpus_name, file_name, number_terms=61, numerical_rep=False, **kwargs):
    """
    Creates a term list for binary documents as a csv file
    Meant to be called from generate_term_list_csv
    :return: None
    """
    print("Generating term,list,freq from binary csv ")
    my_binary = []
    files = glob.glob(corpus_name + "/*")
    ntpath.basename(corpus_name + "/")
    for file in files:
        with open(file, "rb") as f:
            curr = [file]
            count = 0
            while True:
                count += 1
                if numerical_rep:
                    byte = f.read(1)
                else:
                    byte = f.read(1).hex()

                if count > number_terms or not byte:
                    break
                else:
                    if numerical_rep:
                        byte = int.from_bytes(byte, byteorder='big')
                    curr.append(byte)
        my_binary.append(curr)
        # print(curr)

    write_term_list_to_csv(file_name, my_binary, number_terms, **kwargs)


def generate_text_term_list(corpus_name, file_name, number_terms=61, **kwargs):
    """
    Creates a term list of text documents as a csv file
    Meant to be called from generate_term_list_csv
    :return: None
    """
    print("Generating text term list in generic  ")
    my_terms = []
    files = glob.glob(corpus_name + "/*")
    ntpath.basename(corpus_name + "/")
    for file in files:
        with open(file, "rb") as f:
            curr = [file]
            count = 0
            for term in f:
                term = term.decode("utf-8")
                if not term.isspace():

                    if count < number_terms:
                        curr.append(term.strip())
                    else:
                        break
                    count += 1
        my_terms.append(curr)
        # print(curr)
        write_term_list_to_csv(file_name, my_terms, number_terms, **kwargs)


def write_term_list_to_csv(file_name, my_terms, number_terms, transpose=False):
    """
    Writes the terms list to a csv file
    Meant to be called by the generate_term_list files
    :return: None
    """
    print("Writing the term,list into csv ")
    my_csv_rep = []
    header = ["files"]
    header.extend(["byte" + str(i) for i in range(0, number_terms)])
    my_csv_rep.append(header)
    my_csv_rep.extend(my_terms)
    if transpose:
        my_csv_rep = list(map(list, zip(*my_csv_rep)))
    #create the representation
    with open(file_name + ".csv", "w") as w:
        csv_file = csv.writer(w)
        for row in my_csv_rep:
            csv_file.writerow(row)