import pandas as pd
import sys

def generate_igv_status_bed(input_file, output_bed):
    print(f"[*] Loading data from {input_file}...")
    try:
        df = pd.read_csv(input_file, sep='\t')
    except FileNotFoundError:
        print(f"CRITICAL ERROR: {input_file} not found.")
        sys.exit(1)

    # 1. Drop rows missing coordinates to prevent IGV crashing
    df = df.dropna(subset=['chromosome', 'start', 'end'])

    # 2. Format Coordinates for BED standard
    df['start'] = df['start'].astype(int)
    df['end'] = df['end'].astype(int)

    # 3. Construct IGV Metadata Columns
    # Name will display as the primary label next to the track
    df['Name'] = df['gene'] + "_" + df['sgrna sequence']
    
    # Score scaled from ML probability
    df['Score'] = (df['prediction_probability'] * 1000).fillna(0).astype(int)

    # 4. Map the new Color Codes based on gene_status
    def get_status_color(status):
        if status == 'Perfect_Agreement':
            return "0,150,0"     # Green
        elif status == 'Systematic_Mismatch':
            return "200,0,0"     # Red
        elif status == 'Mixed_Performance':
            return "255,140,0"   # Orange
        else:
            return "128,128,128" # Grey (fallback for unexpected values)

    df['itemRgb'] = df['gene_status'].apply(get_status_color)

    # 5. Set boundary columns
    df['thickStart'] = df['start']
    df['thickEnd'] = df['end']

    # 6. Define strict column layout: 9 standard BED columns + 3 extra metadata columns
    # IGV will render the first 9, and display the final 3 when you click the track.
    bed_cols = [
        'chromosome', 'start', 'end', 'Name', 'Score', 'strand',
        'thickStart', 'thickEnd', 'itemRgb',
        'gene_status', 'prediction_binary', 'class'
    ]
    
    # Ensure all target columns exist before slicing
    available_cols = [col for col in bed_cols if col in df.columns]
    final_bed = df[available_cols]

    # 7. Export the BED file
    print(f"[*] Exporting BED file to {output_bed}...")
    
    # Write the IGV track header
    # We update the description so users reading the file know the color key
    track_header = 'track name="sgRNA_Performance" description="Perfect=Green, Mixed=Orange, Mismatch=Red" itemRgb="On"\n'
    
    with open(output_bed, "w") as f:
        f.write(track_header)
    
    # Append the dataframe without pandas headers
    final_bed.to_csv(output_bed, sep='\t', header=False, index=False, mode='a')
    
    print(f"[-->] Operation complete. Generated IGV BED file with {len(final_bed)} tracks.")

if __name__ == "__main__":
    # Ensure these point to the correct files in your directory
    INPUT_FILE = "final_validation_set.txt"
    OUTPUT_BED = "sgrna_performance_status.bed"
    
    generate_igv_status_bed(INPUT_FILE, OUTPUT_BED)
