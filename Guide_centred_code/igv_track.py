import sys
import pandas as pd

def main():
    if len(sys.argv) != 3:
        print("Usage: python generate_igv_window_tracks.py <your_data.txt> <output_guide_windows.bed>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    df = pd.read_csv(input_path, sep='\t')

    print(f"Compiling BED12 multi-block track from matrix data...")
    with open(output_path, "w") as f:
        # Write browser configuration headers for clear track styling
        f.write("track name='Guide_Centered_Windows' description='Calculated 2kb Sliding Extraction Windows' itemRgb='On'\n")
        
        for _, row in df.iterrows():
            chrom = str(row['Chromosome'])
            g_start = int(row['Start'])
            g_end = int(row['End'])
            gene = str(row['Gene'])
            id_val = str(row['ID'])
            strand = str(row['Strand'])
            
            # Extract the actual value calculated for the central landing pad (Bin 5)
            # Your data column: guide_methylation_percent_bin_5
            bin5_score = row['guide_methylation_percent_bin_5']
            display_score = round(bin5_score, 3) if pd.notna(bin5_score) else "0.0"
            
            # Define the global footprint of the entire 2kb window spanned by the bins
            # Bin 1 Start to Bin 10 End
            win_start = int(row['Guide_Bin1_Start'])
            win_end = int(row['Guide_Bin10_End'])
            
            # Define specific sub-blocks inside the window to highlight the guide landing zone (Bin 5 & 6)
            # This allows you to visually track the center of the sliding window in IGV
            b5_start = int(row['Guide_Bin5_Start'])
            b6_end = int(row['Guide_Bin6_End'])
            
            # BED12 Structure calculations
            name = f"ID_{id_val}_{gene}_Pct5:{display_score}"
            score = "1000"
            rgb = "255,127,14" if strand == "+" else "31,119,180" # Orange for +, Blue for -
            
            # Define 3 continuous block segments to render the window shape perfectly:
            # Block 1: Left flank (Bin 1 to 4)
            # Block 2: Target core center (Bin 5 and 6)
            # Block 3: Right flank (Bin 7 to 10)
            block_count = "3"
            
            block_sizes = f"{b5_start - win_start},{b6_end - b5_start},{win_end - b6_end}"
            block_starts = f"0,{b5_start - win_start},{b6_end - win_start}"
            
            # Write standard BED12 row format
            f.write(f"{chrom}\t{win_start}\t{win_end}\t{name}\t{score}\t{strand}\t"
                    f"{win_start}\t{win_end}\t{rgb}\t{block_count}\t{block_sizes}\t{block_starts}\n")

    print(f"Track generation complete. Load file into IGV: {output_path}")

if __name__ == "__main__":
    main()
