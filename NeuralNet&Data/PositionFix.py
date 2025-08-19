import pandas as pd

df = pd.read_excel(r"C:\Users\blake\OneDrive\Desktop\PitchingChange.xlsx")
backup = 500

for index, rows in df.iterrows():
    pos = str(rows['position']) #str is necessary because some positions are already ints
    for i in range(len(pos)):
        try: 
            int(pos[i:i+1])
            df.loc[index, 'position'] = pos[i:i+1] 
            print(pos)
            break
        except:
            if (pos[i:i+1] == "H"):
                df.loc[index, 'position'] = 10
                print(pos)
                break  
            continue
    if index > backup:
        df.to_excel(r"C:\Users\blake\OneDrive\Desktop\PitchingFix.xlsx")
        backup += 500



df.to_excel(r"C:\Users\blake\OneDrive\Desktop\PitchingFix.xlsx")
