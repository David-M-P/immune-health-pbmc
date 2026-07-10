library(progeny)
library(ggplot2)
library(dplyr)
library(tidyverse)

model <- progeny::model_human_full
head(model)

progeny200 <- model %>%
  group_by(pathway) %>%               
  slice_min(order_by = p.value, n = 200) %>%  
  ungroup()                           

write_csv(progeny200, '/lustre/scratch126/cellgen/team361/mm58/gplearner_reproducibility/02_benchmarking/progeny_200_long.csv')

gpdb_dict <- list()

for (c in unique(progeny200$pathway)) {
  gpdb_dict[[c]] <- progeny200[progeny200$pathway == c, "gene"]
}

gpdb <- as.data.frame(gpdb_dict)

colnames(gpdb) <- names(gpdb_dict)

write_csv(gpdb, '/lustre/scratch126/cellgen/team361/mm58/gplearner_reproducibility/02_benchmarking/gpdb_progeny_200.csv')