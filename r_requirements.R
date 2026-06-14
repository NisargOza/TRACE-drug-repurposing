# Run this once to install all R dependencies
CRAN <- "https://cloud.r-project.org"

if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager", repos = CRAN)

BiocManager::install(
    c("limma", "DESeq2", "edgeR", "GEOquery",
      "AnnotationDbi", "huex10sttranscriptcluster.db"),
    ask = FALSE, update = FALSE
)

install.packages(c("metafor", "RobustRankAggreg"), repos = CRAN)

# Mendelian Randomization packages (drug-target MR analysis)
if (!require("remotes", quietly = TRUE))
    install.packages("remotes", repos = CRAN)

remotes::install_github("MRCIEU/TwoSampleMR")
remotes::install_github("mrcieu/ieugwasr")

install.packages(c("coloc", "data.table", "jsonlite", "readr", "dplyr", "ggplot2"),
                 repos = CRAN)
