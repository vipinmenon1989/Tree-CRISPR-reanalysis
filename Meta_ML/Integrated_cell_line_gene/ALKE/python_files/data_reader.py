import csv

file_path = "merged_alke_dataset.csv"

try:
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        row1 = next(reader)
        
        print(f"[*] File opened successfully.")
        print(f"[*] Header found: {header}")
        print(f"[*] First data row: {row1}")
        print(f"[*] Total columns detected: {len(row1)}")
        
        if len(header) != len(row1):
            print("[!] WARNING: Header length does not match data length! This is why your code is crashing.")

except Exception as e:
    print(f"[!] Critical Error: {e}")
