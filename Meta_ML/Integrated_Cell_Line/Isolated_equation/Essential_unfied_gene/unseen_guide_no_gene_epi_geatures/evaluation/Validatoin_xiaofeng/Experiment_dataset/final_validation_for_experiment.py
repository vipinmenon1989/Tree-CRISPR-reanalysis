import pandas as pd

# 1. Load the comprehensive data
df = pd.read_csv("Final_Validated_Dataset.txt", sep='\t')

# ---------------------------------------------------------
# TASK 1: Clean and format the main manifest
# ---------------------------------------------------------
manifest_df = df.copy()

# Drop the 'id' column
if 'id' in manifest_df.columns:
    manifest_df = manifest_df.drop(columns=['id'])

# Rename the status column
manifest_df = manifest_df.rename(columns={'gene_status': 'Status (Compared to Ground truth)'})

# Convert binary predictions to High/Low strings
manifest_df['prediction_binary'] = manifest_df['prediction_binary'].replace({1: 'High', 0: 'Low'})

# Save the polished main file
manifest_df.to_csv("ALKE_Lab_Validation.txt", sep='\t', index=False)


# ---------------------------------------------------------
# TASK 2: Generate the TSS BED File
# ---------------------------------------------------------
# Extract unique TSS coordinates for the genes in this set
tss_df = df[['Chromosome', 'closest_TSS_coord', 'gene', 'Strand']].drop_duplicates()
tss_df = tss_df.dropna(subset=['closest_TSS_coord'])

# BED format requires integers. A single TSS point is represented as Start to Start+1
tss_df['Start'] = tss_df['closest_TSS_coord'].astype(int)
tss_df['End'] = tss_df['Start'] + 1
tss_df['Name'] = tss_df['gene'] + "_TSS"
tss_df['Score'] = 0  # Standard placeholder for BED score

tss_bed = tss_df[['Chromosome', 'Start', 'End', 'Name', 'Score', 'Strand']]

# Save TSS BED file
tss_bed.to_csv("Gene_TSS_Coordinates.bed", sep='\t', header=False, index=False)


# ---------------------------------------------------------
# TASK 3: Generate the Color-Coded sgRNA BED File
# ---------------------------------------------------------
sgrna_bed = df.copy()

# Name the BED element using Gene and Sequence for easy identification
sgrna_bed['Name'] = sgrna_bed['gene'] + "_" + sgrna_bed['sgrna sequence']

# Scale prediction probability to a BED score (0-1000) for visual shading if needed
sgrna_bed['Score'] = (sgrna_bed['prediction_probability'] * 1000).astype(int)

# Map the binary prediction to an RGB color string (Green for 1/High, Red for 0/Low)
def get_rgb_color(binary_val):
    if binary_val == 1:
        return "0,180,0"   # Dark Green
    else:
        return "200,0,0"   # Red

sgrna_bed['itemRgb'] = sgrna_bed['prediction_binary'].apply(get_rgb_color)

# BED9 format requires thickStart and thickEnd (we just match the sequence boundaries)
sgrna_bed['thickStart'] = sgrna_bed['Start']
sgrna_bed['thickEnd'] = sgrna_bed['End']

# Select the strict 9 columns required for RGB coloring
bed9_columns = [
    'Chromosome', 'Start', 'End', 'Name', 'Score', 'Strand', 
    'thickStart', 'thickEnd', 'itemRgb'
]
final_sgrna_bed = sgrna_bed[bed9_columns]

# Save sgRNA BED file with a track header so genome browsers read the colors
with open("sgRNA_Predictions_ColorCoded.bed", "w") as f:
    # This header instructs the browser to parse the 9th column as a color
    f.write('track name="sgRNA Predictions" description="High=Green, Low=Red" itemRgb="On"\n')
    
final_sgrna_bed.to_csv("sgRNA_Predictions_ColorCoded.bed", sep='\t', header=False, index=False, mode='a')

print("--- FILE GENERATION COMPLETE ---")
print("1. Final_Lab_Manifest.txt (ID dropped, columns renamed, High/Low logic applied)")
print("2. Gene_TSS_Coordinates.bed (Standard 6-column BED)")
print("3. sgRNA_Predictions_ColorCoded.bed (9-column BED with RGB values applied)")
