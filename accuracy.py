def findingAccuracy_shakespeare(file_names, labels_predicted):
    president_data = 0
    shakespeare_data = 0
    president_data_correct = 0
    shakespeare_data_correct = 0

    for i in range(0, len(file_names)):
        filename = file_names[i]
        str = filename[1:4]
        if RepresentsInt(str):
            president_data = president_data + 1
            if labels_predicted[i] == 0:
                president_data_correct = president_data_correct + 1
        else:
            shakespeare_data = shakespeare_data + 1
            if labels_predicted[i] == 1:
                shakespeare_data_correct = shakespeare_data_correct + 1

    print("Number of president text files are", president_data)
    print("Number of president text files predicted correctly is", president_data_correct)
    print("Accuracy : ", (president_data_correct / president_data) * 100)

    print("Number of shakespeare text files are", shakespeare_data)
    print("Number of shakespeare text files predicted correctly is", shakespeare_data_correct)
    print("Accuracy : ", (shakespeare_data_correct / shakespeare_data) * 100)


def findingAccuracy_malware(file_names, labels_predicted):
    MaliciousData = 0
    BenignData = 0
    MaliciousDataCorrect = 0
    BenignDataCorrect = 0

    for i in range(0, len(file_names)):
        filename = file_names[i]
        str = filename[0:4]
        if str == "zeus":
            MaliciousData = MaliciousData + 1
            if labels_predicted[i] == 0:
                MaliciousDataCorrect = MaliciousDataCorrect + 1
        else:
            BenignData = BenignData + 1
            if labels_predicted[i] == 1:
                BenignDataCorrect = BenignDataCorrect + 1

    print("Number of malicious text files are", MaliciousData)
    print("Number of malicious text files predicted correctly is", MaliciousDataCorrect)
    print("Accuracy : ", (MaliciousDataCorrect / MaliciousData) * 100)

    print("Number of benign text files are", BenignData)
    print("Number of benign text files predicted correctly is", BenignDataCorrect)
    print("Accuracy : ", (BenignDataCorrect / BenignData) * 100)


def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
