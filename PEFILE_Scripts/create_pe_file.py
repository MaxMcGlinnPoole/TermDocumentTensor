import pefile
import os
import time

class PEFile():
    def __init__(self, file_location, file):
        self.file = file
        self.file_header_characteristics = ["IMAGE_FILE_BYTES_REVERSED_HI", "IMAGE_FILE_DLL",
                                            "IMAGE_FILE_DEBUG_STRIPPED", "IMAGE_FILE_LARGE_ADDRESS_AWARE",
                                            "IMAGE_FILE_LOCAL_SYMS_STRIPPED","IMAGE_FILE_RELOCS_STRIPPED",
                                            "IMAGE_FILE_LINE_NUMS_STRIPPED"]
        self.sections_flags = ["IMAGE_SCN_CNT_UNINITIALIZED_DATA", "IMAGE_SCN_ALIGN_1BYTES",
                                      "IMAGE_SCN_ALIGN_2BYTES", "IMAGE_SCN_ALIGN_4BYTES", "IMAGE_SCN_MEM_DISCARDABLE",
                                      "IMAGE_SCN_MEM_NOT_PAGED", "IMAGE_SCN_MEM_WRITE", "IMAGE_SCN_MEM_SHARED"]
        self.file_information = []

        self.load_from_file(file_location)


    def load_from_file(self, file_location):
        try:
            pe = pefile.PE(file_location)
        except Exception as e:
            print(file_location)
            print(e)
            return
        file_header = pe.NT_HEADERS.FILE_HEADER.__dict__
        opt_header = pe.OPTIONAL_HEADER.__dict__
        timedate = file_header["TimeDateStamp"]

        functions = []
        dlls = []
        characteristics = []
        sections_flags = []
        times = []
        sizes = []
        various = []
        timedate = time.localtime(timedate)

        if timedate.tm_year < 1992:
            times.append("before")
        elif timedate.tm_year > 2015:
            times.append("after")
        else:
            times.append("during")
        times.append(str(timedate.tm_hour))


        for section in pe.sections:
            section = section.__dict__
            for flag in self.sections_flags:
                if section[flag]:
                    sections_flags.append(flag)
        if len(sections_flags) == 0:
            sections_flags.append("NONE")
        for flag in self.file_header_characteristics:
            if file_header[flag]:
                characteristics.append(flag)
        if file_header['NumberOfSections'] > 8 or file_header['NumberOfSections'] < 1:
            various.append("sections_abnormal")
        else:
            various.append("sections_normal")
        if file_header['PointerToSymbolTable'] > 0:
            various.append("debugging_info")

        if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                dlls.append(entry.dll.decode("utf-8"))
                for imp in entry.imports:
                    try:
                        functions.append(imp.name.decode("utf-8"))
                    except:
                        pass

        else:
            functions.append("NONE")
            dlls.append("NONE")
        size = os.stat(file_location).st_size
        code_ratio = opt_header["SizeOfCode"]/size
        initial_ratio = opt_header["SizeOfInitializedData"]/size
        uninitial_ratio = opt_header["SizeOfUninitializedData"]/size
        image_ratio = opt_header["SizeOfImage"]/size
        header_ratio = opt_header["SizeOfHeaders"]/size
        if code_ratio > 1:
            sizes.append("code_abnormal")
        if initial_ratio > 3:
            sizes.append("initial_abnormal")
        if uninitial_ratio > 1:
            sizes.append("uninitial_abnormal")
        if image_ratio > 8:
            sizes.append("image_abnormal")
        if header_ratio > 0:
            sizes.append("header_abnormal")


        self.file_information.append(functions)
        self.file_information.append(dlls)
        self.file_information.append(characteristics)
        self.file_information.append(sections_flags)
        self.file_information.append(times)
        self.file_information.append(sizes)
        self.file_information.append(various)

    def write_to_file(self, save_path):
        completePath = os.path.join(save_path, self.file + ".txt")
        with open(completePath, "w") as write:
            for entry in self.file_information:
                write.write(" ".join(entry) + "\n")

def main():
    save_path = "output"
    directories = ["benign", "test_files"]
    for directory in directories:
        files = os.listdir(directory)
        for file in files:
            pe_file = PEFile(directory + "/" + file, file)
            pe_file.write_to_file(save_path)

main()