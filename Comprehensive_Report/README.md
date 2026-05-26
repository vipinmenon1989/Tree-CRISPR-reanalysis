# Comprehensive_Report — Cross-Cell-Line Summary

Generates a comprehensive comparative plot across all cell lines and conditions.

## Contents

| Script | Purpose |
|--------|---------|
| `plot_comprehensive.py` | Produce a unified summary plot comparing hit genes across A375, A549, and K562 screens |

## Usage

```bash
python plot_comprehensive.py
```

Expects merged hit tables from each cell line's reanalysis in the parent directory or specified paths inside the script.

## Output

A single comprehensive figure summarising resistance hits, essentiality scores, and sgRNA-level confidence across all conditions and cell lines.
