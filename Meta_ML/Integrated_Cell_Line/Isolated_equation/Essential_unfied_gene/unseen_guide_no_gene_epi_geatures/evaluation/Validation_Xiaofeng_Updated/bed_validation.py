import pandas as pd
import numpy as np

def generate_bed_files(input_path, tss_bed_path, sgrna_bed_path):
    # Load the dataset
    df = pd.read_csv(input_path, sep="\t")
    
    # Strip invisible whitespace from all column names
    df.columns = df.columns.str.strip()
    
    # Drop rows missing crucial genomic coordinates to prevent integer casting errors
    df = df.dropna(subset=['Chromosome', 'Start', 'End', 'closest_TSS_coord'])
    
    # ---------------------------------------------------------
    # 1. GENERATE THE SGRNA BED FILE (Colored)
    # ---------------------------------------------------------
    
    # Define the strict logic boundaries
    cond_high = (df['prediction_probability'] > 0.5) & (df['sigmoid_score'] > 0.25)
    cond_low = (df['prediction_probability'] < 0.5) & (df['sigmoid_score'] <= 0.05)
    
    # Apply colors based on conditions:
    # Green (0,255,0) for High, Red (255,0,0) for Low. 
    # Ambiguous data in the deadzone defaults to Gray (128,128,128).
    colors = np.select([cond_high, cond_low], ['0,255,0', '255,0,0'], default='128,128,128')
    
    # Construct the BED9 standard dataframe
    # thickStart and thickEnd must equal chromStart and chromEnd for solid block coloring
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
        f.write('track name="sgRNA_Predictions" description="High=Green, Low=Red" itemRgb="On"\n')
    
    # Append the dataframe data (no headers, tab-separated)
    df_sgrna.to_csv(sgrna_bed_path, sep="\t", index=False, header=False, mode='a')
    
    # ---------------------------------------------------------
    # 2. GENERATE THE TSS BED FILE
    # ---------------------------------------------------------
    
    # Construct the TSS dataframe
    # BED features require length. We add +1 to the TSS coordinate to create a 1-bp region.
    df_tss = pd.DataFrame({
        'chrom': df['Chromosome'],
        'chromStart': df['closest_TSS_coord'].astype(int),
        'chromEnd': (df['closest_TSS_coord'] + 1).astype(int),
        'name': df['gene'],
        'score': 0,
        'strand': df['Strand']
    })
    
    # Since multiple guides target the same gene, deduplicate to keep the TSS BED clean
    df_tss = df_tss.drop_duplicates(subset=['chrom', 'chromStart', 'name'])
    
    # Write the TSS BED file
    with open(tss_bed_path, 'w') as f:
        f.write('track name="Target_TSS" description="Transcription Start Sites" color="0,0,255"\n')
    df_tss.to_csv(tss_bed_path, sep="\t", index=False, header=False, mode='a')
    
    # Logic check
    print(f"Total processed guides: {len(df_sgrna)}")
    print(f"Total unique TSS sites extracted: {len(df_tss)}")
    print(f"High (Green) sgRNAs: {cond_high.sum()}")
    print(f"Low (Red) sgRNAs: {cond_low.sum()}")
    print(f"Files successfully generated:\n  - {sgrna_bed_path}\n  - {tss_bed_path}")

# Example execution:
generate_bed_files('Validation_filtered_vipin_list.txt', 'TSS_coordinates.bed', 'sgRNA_colored.bed')
