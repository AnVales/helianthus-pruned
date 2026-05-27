# Population Genomic Structure Analysis in Helianthus annuus

This project explores genetic variation in Helianthus annuus using whole-genome SNP data. A full pipeline was implemented, including variant filtering, LD pruning, PCA, and clustering validation.

## Methods

- SNP filtering (bcftools, vcftools)
- LD pruning (PLINK)
- PCA (PLINK)
- Clustering (KMeans)
- Validation (Silhouette Score)

## Key Results

- Clear population structure detected
- Silhouette score: 0.78 (strong clustering)
- Evidence of substructure within clusters

## Tools

- PLINK
- bcftools
- vcftools
- Python (pandas, sklearn, matplotlib)
