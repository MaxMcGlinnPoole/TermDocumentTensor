import pefile
import os
import time
def main():
    save_path = "output"
    directory = "test_files"
    files = os.listdir(directory)
    count = 0
    hours = [0]*24
    years = [0]*(2018-1970)
    for file in files:
        if count == 100:
            break

        try:
            pe = pefile.PE(directory + "/" + file)
        except Exception as e:
            print(e)
        timedate = pe.FILE_HEADER.TimeDateStamp
        opt_header = pe.OPTIONAL_HEADER.dump_dict()

        functions = []
        dlls = []
        characteristics = []
        warnings = []
        times = []
        timedate = time.localtime(timedate)
        if timedate.tm_year < 1992:
            times.append("before")
        elif timedate.tm_year > 2015:
            times.append("after")
        else:
            times.append("during")
        times.append(str(timedate.tm_hour))
        hours[timedate.tm_hour] += 1
        file_header = pe.NT_HEADERS.FILE_HEADER
        if file_header.IMAGE_FILE_BYTES_REVERSED_HI:
            characteristics.append("IMAGE_FILE_BYTES_REVERSED_HI")
        if file_header.IMAGE_FILE_DLL:
            characteristics.append("IMAGE_FILE_DLL")
        if file_header.IMAGE_FILE_LARGE_ADDRESS_AWARE:
            characteristics.append("IMAGE_FILE_LARGE_ADDRESS_AWARE")
        if file_header.IMAGE_FILE_LOCAL_SYMS_STRIPPED:
            characteristics.append("IMAGE_FILE_LOCAL_SYMS_STRIPPED")
        if file_header.IMAGE_FILE_RELOCS_STRIPPED:
            characteristics.append("IMAGE_FILE_RELOCS_STRIPPED")
        if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                dlls.append(entry.dll.decode("utf-8"))
                for imp in entry.imports:
                    try:
                        functions.append(imp.name.decode("utf-8"))
                    except:
                        print("Problem child ", imp.name)
        else:
            functions.append("NONE")
            dlls.append("NONE")
        if hasattr(pe, "DIRECTORY_ENTRY_EXPORT"):
            for exp in pe.DIRECTORY_ENTRY_EXPORT.symbols:
                print( exp.name, exp.ordinal)
        if hasattr(pe, "_PE_warnings"):
            for warning in pe._PE_warnings:
                warnings.append(warning)
        else:
            warnings.append("NONE")

        completePath = os.path.join(save_path, file + ".txt")
        with open(completePath, "w") as write:
            write.write(" ".join(functions) + "\n")
            write.write(" ".join(dlls) + "\n")
            write.write(" ".join(times) + "\n")
            write.write(" ".join(characteristics))
        count+=1
    """hours = [float(i) / sum(hours) for i in hours]
    years = [float(i) / sum(years) for i in years]
    print(hours)
    print(years)"""



main()