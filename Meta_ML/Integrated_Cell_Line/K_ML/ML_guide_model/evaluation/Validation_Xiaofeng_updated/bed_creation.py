import pandas as pd
import numpy as np

def generate_validation_bed_files(input_path, tss_bed_path, sgrna_bed_path):
    # Load the dataset
    df = pd.read_csv(input_path, sep="\t")
    
    # Strip invisible whitespace from all column names
    df.columns = df.columns.str.strip()
    
    # Drop rows missing crucial genomic coordinates to prevent integer casting errors
    df = df.dropna(subset=['Chromosome', 'Start', 'End', 'closest_TSS_coord'])
    
    # ---------------------------------------------------------
    # 1. GENERATE THE SGRNA BED FILE (Colored)
    # ---------------------------------------------------------
    
    # Apply colors based strictly on the 'Effectiveness' column:
    # Green (0,255,0) for High, Red (255,0,0) for Low.
    # .str.strip().str.upper() ensures case-insensitivity and ignores accidental whitespace.
    colors = np.where(df['Effectiveness'].str.strip().str.upper() == 'HIGH', '0,255,0', '255,0,0')
    
    # Construct the BED9 standard dataframe
    # thickStart and thickEnd are mapped to Start and End to color the entire sequence block
    df_sgrna = pd.DataFrame({
        'chrom': df['Chromosome'],
        'chromStart': df['Start'].astype(int),
        'chromEnd': df['End'].astype(int),
        'name': df['unique_sgrna_id'],
        'score': 0, # Standard BED filler score
        'strand': df['Strand'],
        'thickStart': df['Start'].astype(int),
        'thickEnd': df['End'].astype(int),
        'itemRgb': colors
    })
    
    # Write the sgRNA BED file with the required IGV track configuration line
    with open(sgrna_bed_path, 'w') as f:
        f.write('track name="sgRNA_Validation" description="High=Green, Low=Red" itemRgb="On"\n')
    
    # Append the dataframe data (no headers, tab-separated)
    df_sgrna.to_csv(sgrna_bed_path, sep="\t", index=False, header=False, mode='a')
    
    # ---------------------------------------------------------
    # 2. GENERATE THE TSS BED FILE
    # ---------------------------------------------------------
    
    # Construct the TSS dataframe
    # BED coordinate systems require length, so we add +1 to the closest_TSS_coord to create a 1-bp feature
    df_tss = pd.DataFrame({
        'chrom': df['Chromosome'],
        'chromStart': df['closest_TSS_coord'].astype(int),
        'chromEnd': (df['closest_TSS_coord'] + 1).astype(int),
        'name': df['gene'],
        'score': 0,
        'strand': df['Strand']
    })
    
    # Deduplicate to ensure only one TSS feature is drawn per gene
    df_tss = df_tss.drop_duplicates(subset=['chrom', 'chromStart', 'name'])
    
    # Write the TSS BED file
    with open(tss_bed_path, 'w') as f:
        f.write('track name="Validation_TSS" description="Transcription Start Sites" color="0,0,255"\n')
    df_tss.to_csv(tss_bed_path, sep="\t", index=False, header=False, mode='a')
    
    # Logic check
    print(f"Total processed validation guides: {len(df_sgrna)}")
    print(f"Total unique TSS sites extracted: {len(df_tss)}")
    print(f"High (Green) sgRNAs: {(colors == '0,255,0').sum()}")
    print(f"Low (Red) sgRNAs: {(colors == '255,0,0').sum()}")
    print(f"Files successfully generated:\n  - {sgrna_bed_path}\n  - {tss_bed_path}")

generate_validation_bed_files('Validation_Xiaofeng_vipin_list.txt', 'Xiaofeng_TSS.bed', 'Xiaofeng_sgRNA.bed')
