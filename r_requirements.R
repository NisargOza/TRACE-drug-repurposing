# Run this once to install all R dependencies
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install(c("limma", "DESeq2", "edgeR", "GEOquery"))

install.packages(c("metafor", "RobustRankAggreg"))
