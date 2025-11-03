# Inversion with quantile transform

library(KScorrect)
library(mclust)

set.seed(311025)

current_directory <- getwd()

experiment<- "GMCM"
#experiment<- "wisconsin"

output_parameters_path <- file.path(current_directory,"imputation",experiment,"output_parameters")
weights_matrix_output <- as.matrix(read.csv(file.path(output_parameters_path,"weights_matrix_output.csv"), header = FALSE))
means_matrix_output <- as.matrix(read.csv(file.path(output_parameters_path,"means_matrix_output.csv"), header = FALSE))
sds_matrix_output <- as.matrix(read.csv(file.path(output_parameters_path,"sds_matrix_output.csv"), header = FALSE))

before_transfo <- file.path("imputation",experiment,"before_transfo")

conditional_parameters <- file.path("imputation",experiment,"conditional_parameters")
# Get the current working directory
current_directory <- getwd()

# Construct the new directory path
before_transfo_path <- file.path(current_directory, before_transfo)

conditional_parameters_path <- file.path(current_directory, conditional_parameters)


files <- list.files(path = folder_path)
number_of_files <- length(files)
n_observations = number_of_files
results_quantile_transform = list()
start = Sys.time()
for (i in 1:n_observations){
  samples_matrix = as.matrix(read.csv(file.path(before_transfo_path,paste0("array_",i-1,".csv")), header = TRUE))
  means_matrix_input = as.matrix(read.csv(file.path(conditional_parameters_path,paste0("element_",i-1,"/means.csv")), header = FALSE))
  sds_matrix_input = as.matrix(read.csv(file.path(conditional_parameters_path,paste0("element_",i-1,"/stds.csv")), header = FALSE))
  weights_matrix_input = as.matrix(read.csv(file.path(conditional_parameters_path,paste0("element_",i-1,"/weights.csv")), header = FALSE))
  indexes = as.matrix(read.csv(file.path(conditional_parameters_path,paste0("element_",i-1,"/indexes.csv")), header = FALSE))
  n_columns = ncol(samples_matrix)
  n_rows = nrow(samples_matrix)
  transformed_samples = matrix(data=NA, nrow = n_rows,ncol= n_columns)
  for (j in 1:n_columns){
    m = indexes[j]+1
    transformed_samples[,j] = KScorrect::qmixnorm(p=KScorrect::pmixnorm(q=samples_matrix[,j],
                                                                        mean = means_matrix_input[means_matrix_input[, m] != 0, m], 
                                                                        sd=sds_matrix_input[sds_matrix_input[, m] != 0, m],
                                                                        pro=weights_matrix_input), 
                                                  mean =means_matrix_output[means_matrix_output[, m] != 0, m]
                                                  , sd =sds_matrix_output[sds_matrix_output[, m] != 0, m]
                                                  , pro= weights_matrix_output[weights_matrix_output[, m] != 0, m], expand=0 )
  }
  results_quantile_transform[[i]] = transformed_samples
  print(paste0("done_",i))
}
end = Sys.time()

afer_transfo <- file.path(current_directory, "imputation",experiment,"after_transfo")
csv_files <- list.files(pattern = "\\.csv$")
file.remove(csv_files)
lapply(seq_along(results_quantile_transform), function(i) {
  write.csv(results_quantile_transform[[i]], file.path(after_transfo,paste0("matrix", i, ".csv")), row.names = FALSE)
})