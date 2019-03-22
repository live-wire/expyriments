# Pandas read xlsx

import numpy as np
import pandas as pd
import os

xlsx_path = "dl_xlsx"
md_path = "dl_md"
xlsx_files = os.listdir(xlsx_path)
xlsx_files = list(filter(lambda x: ".DS_Store" not in x, xlsx_files))

def np_to_md(np_arr, filename):
    filename = filename[:filename.rfind(".")]
    fout = os.path.join(md_path, filename+".md")

    str_out = "# Questions from `%s` :robot: \n\n"%filename
    for item in np_arr: 
        for i, it in enumerate(item):
            l = "- "
            lback = ""
            if i==0:
                continue
            elif i==1:
                l = "**Q: "
                lback = "**"
            elif i==6:
                l = "* `[ "
                lback = " ]`"
            str_out += l+str(it).strip("\n")+ lback +"\n"
        str_out += "\n\n---\n\n"
    with open(fout, "w") as f:
        f.write(str_out)
    print("Generated:", fout)


for filename in xlsx_files:
    f = os.path.join(xlsx_path, filename)
    print("Processing:", f)
    df = pd.read_excel(xlsx_path+"/"+filename, error_bad_lines = False)
    #print(df)
    np_arr = df.to_records()
    np_to_md(np_arr, filename)
