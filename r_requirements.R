# Run this once to install all R dependencies
CRAN <- "https://cloud.r-project.org"

user_lib <- file.path(getwd(), "Rlib")
dir.create(user_lib, showWarnings = FALSE, recursive = TRUE)
.libPaths(c(user_lib, .libPaths()))

if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager", repos = CRAN, lib = user_lib)

BiocManager::install(
    c("limma", "DESeq2", "edgeR", "GEOquery",
      "AnnotationDbi", "huex10sttranscriptcluster.db"),
    ask = FALSE, update = FALSE, lib = user_lib
)

install.packages(c("metafor", "RobustRankAggreg"), repos = CRAN, lib = user_lib)

# Mendelian Randomization packages (drug-target MR analysis)
if (!require("remotes", quietly = TRUE))
    install.packages("remotes", repos = CRAN, lib = user_lib)

remotes::install_github("MRCIEU/TwoSampleMR", lib = user_lib)
remotes::install_github("mrcieu/ieugwasr", lib = user_lib)

install.packages(c("coloc", "data.table", "jsonlite", "dplyr", "ggplot2"),
                 repos = CRAN, lib = user_lib)