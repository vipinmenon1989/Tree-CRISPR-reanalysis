import sys
import pandas as pd
import os

def main():
    if len(sys.argv) < 3:
        print("Error: Missing input arguments.")
        print("Usage: python convert_to_colored_bed.py <input_csv_file> <output_bed_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)

    try:
        # Load the input file
        df = pd.read_csv(input_file)
        
        # Standardize column names
        df.columns = [str(c).strip() for c in df.columns]

        # Construct name column with relevant metadata for IGV
        df['name'] = (
            df['ID'].astype(str) + "_" +
            df['Gene'].astype(str) + "_" +
            df['sgRNA Sequence'].astype(str) + "_TSS" +
            df['distance_to_TSS'].astype(str)
        )
        
        df['score'] = 1000

        # Assign colors based on distance to TSS
        def get_rgb(dist):
            if dist <= 0:
                return "31,119,180"  # Blue for non-positive distance
            else:
                return "214,39,40"   # Red for positive distance

        df['itemRgb'] = df['distance_to_TSS'].apply(get_rgb)

        # Build BED9 DataFrame required for itemRgb
        bed_df = pd.DataFrame({
            'chrom': df['Chromosome'],
            'chromStart': df['Start'],
            'chromEnd': df['End'],
            'name': df['name'],
            'score': df['score'],
            'strand': df['Strand'],
            'thickStart': df['Start'],
            'thickEnd': df['End'],
            'itemRgb': df['itemRgb']
        })

        # Save to a tab-delimited text file with track line header
        with open(output_file, 'w') as f:
            f.write('track name="sgRNA_Distance" description="sgRNA Distance to TSS" itemRgb="On"\n')
            bed_df.to_csv(f, sep='\t', index=False, header=False)
        
        print(f"Process complete. Converted {len(bed_df)} rows to BED format and saved to '{output_file}'.")

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
