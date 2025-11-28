import json
import datetime

print("Started Reading JSON file")
with open("concepts_nb.ipynb", "r") as read_file:
    print("Converting JSON encoded data into Python dictionary")
    note = json.load(read_file)

    print("Decoded JSON Data From File")
    for c in note["cells"]:
        if c["cell_type"] == "code":
            s = c["metadata"]["execution"]["iopub.execute_input"]
            e = c["metadata"]["execution"]["shell.execute_reply"]
            s = s.split("T")[1].split("Z")[0]
            e = e.split("T")[1].split("Z")[0]
            s = datetime.datetime.strptime(s,"%H:%M:%S.%f")
            e = datetime.datetime.strptime(e,"%H:%M:%S.%f")
            print(e," ", s, "\t", e-s)
            # for key, value in c.items():
            #     print(key)
    print("Done reading json file")
