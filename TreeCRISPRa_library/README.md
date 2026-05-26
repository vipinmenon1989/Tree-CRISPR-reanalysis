# TreeCRISPRa_library — CRISPRa Library Epigenetic Extraction

Epigenetic feature extraction for the Tree CRISPRa guide RNA library.

## Contents

| Script | Purpose |
|--------|---------|
| `epigenetic_extraction.py` | Extract BigWig epigenetic signals (Mean, Max, Std per bin) across the CRISPRa library sgRNA genomic windows |

## Usage

```bash
python epigenetic_extraction.py --library <crispra_library.txt> --bigwig_dir <bw_folder/> --output <matrix.txt>
```

This script is the CRISPRa-specific version of the general `epigenetic_pipeline/epigenetic_extraction.py`. It is optimised for the CRISPRa library coordinate format.

## Input

- CRISPRa library file with genomic coordinates
- Directory of BigWig files (e.g. ATAC-seq, H3K27ac, H3K4me3)

## Output

- Feature matrix: one row per sgRNA, columns = epigenetic signal bins per track
