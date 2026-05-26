# TreeCRISPRi_library — CRISPRi Library Processing

Scripts for processing the Tree CRISPRi guide RNA library — including sgRNA mapping, BED file generation, TSS overlap, and epigenetic feature extraction.

## Contents

| Script | Purpose |
|--------|---------|
| `alke_sgrna.py` | Extract ALK-E condition sgRNAs from the CRISPRi library |
| `bedmaker.py` | Convert CRISPRi library coordinates to BED format |
| `epigenetic_extraction.py` | Extract BigWig epigenetic signals for CRISPRi library sgRNAs |
| `overlap.py` | Compute overlap between sgRNA genomic positions and epigenetic feature regions |

## Usage

```bash
# Step 1: Generate BED file from library
python bedmaker.py <library.txt> <output.bed>

# Step 2: Extract epigenetic features
python epigenetic_extraction.py --library <library.txt> --bigwig_dir <bw_folder/> --output <matrix.txt>

# Step 3: Compute overlaps with feature regions
python overlap.py <sgrna.bed> <features.bed> <output.txt>

# Step 4: Extract ALK-E specific sgRNAs
python alke_sgrna.py <library.txt> <output.txt>
```

## Input

- CRISPRi library file with gene, sgRNA sequence, and coordinates
- BigWig files for epigenetic signals
- BED files for genomic feature regions (e.g. enhancers, promoters)

## Output

- BED file of sgRNA positions
- Epigenetic feature matrix
- Overlap summary table
