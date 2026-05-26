# epigenetic_pipeline — Genomic Coordinate & Epigenetic Feature Extraction

This folder contains the end-to-end pipeline for mapping sgRNA genomic coordinates, computing TSS distances, extracting epigenetic signals from BigWig files, binning signals across TSS windows, and generating IGV-compatible BED files.

## Pipeline Steps (Run in Order)

```
Step 0: mapper.py              → Map sgRNA sequences to genome (Bowtie)
Step 0: tss_to_bed.py          → Convert TSS annotation CSV to BED
Step 1: calc_distance_tss_revised.py  → Calculate sgRNA distance to nearest TSS
Step 1: diagnose.py            → Diagnose missing genes before Step 1
Step 2: windomaker.py          → Create 200bp bins across ±1kb TSS window
Step 2: binning_files.py       → Alternative binning utility
Step 3a: all_bigwig.py         → Extract signal from all BigWig tracks in a folder
Step 3b: specific_bigwig.py    → Extract raw coverage + CpG methylation (BigWig + BigBed)
Step 4: epigenetic_extraction.py → Parallel multi-BigWig extraction (Mean/Max/Std per bin)
Step 5: NaN_remover.py         → Clean NaN values from the final matrix
Step 6: igv_maker_analysis.py  → Generate IGV-ready BED files from the matrix
Step 7: matrix_to_bed.py       → Convert horizontal bin matrix to long-form BED
Step 8: analysis.py            → Merge and sort hit files for comprehensive reporting
```

## Scripts

### Step 0 — Coordinate Mapping

**`mapper.py`** — Maps sgRNA sequences from a CSV to the genome using Bowtie 1. Produces a TSV with chromosome, start, end, and strand for each guide.
```bash
python mapper.py <library.csv> <bowtie_index_prefix> <output.tsv>
```

**`tss_to_bed.py`** — Converts a TSS annotation CSV (with `Primary TSS, 5'` column) to a standard BED file.
```bash
python tss_to_bed.py <tss_annotation.csv> <output.bed>
```

**`Extend_sgRNA.py`** — Extends sgRNA genomic coordinates upstream/downstream and fetches the extended sequence from a reference FASTA using pyfaidx. Useful for PAM extraction or context sequence retrieval.
```bash
python Extend_sgRNA.py  # Edit input paths inside the script
```

### Step 1 — TSS Distance

**`calc_distance_tss_revised.py`** *(recommended)* — Calculates the distance from each sgRNA to its gene's primary TSS (P1 transcript). Handles alternative promoters and strand-aware distance.
```bash
python calc_distance_tss_revised.py <library.txt> <tss_annotation.csv> <step1_output.txt>
```

**`calc_distance_tss.py`** — Original version; uses all TSS isoforms (not just P1).

**`calc_distance_tss_P1.py`** — P1-only variant with additional file existence validation.

**`diagnose.py`** — Run before Step 1 to identify genes in the library that are missing from the TSS annotation, preventing silent data loss.
```bash
python diagnose.py  # Edit lib_path and tss_path inside the script
```

### Step 2 — Windowing

**`windomaker.py`** — Takes the Step 1 output and creates a 10-bin matrix of 200bp windows spanning ±1kb around each TSS.
```bash
python windomaker.py <step1_output.txt> <step2_output.txt>
```

**`binning_files.py`** — Alternative binning utility that generates individual bin rows; useful for custom window sizes.

### Step 3 — BigWig Signal Extraction

**`all_bigwig.py`** — Extracts mean signal from every `.bw` file in a specified folder across the Step 2 bins. Produces a unified feature matrix.
```bash
python all_bigwig.py <step2_output.txt> <bigwig_folder/> <final_matrix.txt>
```

**`specific_bigwig.py`** — Extracts both raw coverage (BigWig) and CpG methylation (BigBed) for dual-track methylation analysis.
```bash
python specific_bigwig.py <step2_output.txt> <signal.bw> <cpg.bigBed> <output.txt>
```

**`epigenetic_extraction.py`** — Parallel multi-BigWig extractor using `ProcessPoolExecutor`. Extracts Mean, Max, and Std per bin per track across all guides simultaneously. Best for large BigWig collections.
```bash
python epigenetic_extraction.py --library <lib.txt> --bigwig_dir <bw_folder/> --output <matrix.txt>
```

### Step 4–8 — Cleanup & Visualisation

**`NaN_remover.py`** — Removes rows with NaN values from the feature matrix.
```bash
python NaN_remover.py <matrix_file.txt>
```

**`igv_maker_analysis.py`** — Generates three IGV-compatible BED files from the matrix: guide target positions, coverage bins, and percent-signal bins.
```bash
python igv_maker_analysis.py <input_matrix.txt>
```

**`matrix_to_bed.py`** — Converts the horizontal bin matrix to long-form BED format for genome browser tracks.
```bash
python matrix_to_bed.py <step2_output.txt> <output.bed>
```

**`analysis.py`** — Merges hit files from multiple conditions in a defined order (K → ALKE → ALKL → ALKH → ALKME → ALKMH → ALKML) for comprehensive reporting.

## Input Files Required

| File | Description |
|------|-------------|
| `library.txt` / `library.csv` | sgRNA library with `Gene`, `sgRNA`, `Sequence` columns |
| `tss_annotation.csv` | TSS annotation with `Primary TSS, 5'`, `gene`, `strand`, `transcript` columns |
| `*.bw` | BigWig epigenetic signal files (ATAC-seq, H3K27ac, methylation, etc.) |
| Bowtie index | Pre-built Bowtie 1 genome index |
| Reference FASTA | hg38 or equivalent (for `Extend_sgRNA.py`) |

## Dependencies

```bash
pip install pandas numpy pyBigWig pyfaidx
conda install -c bioconda bowtie
```
