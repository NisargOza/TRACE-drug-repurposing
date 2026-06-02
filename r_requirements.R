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
