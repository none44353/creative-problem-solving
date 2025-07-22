#Use this script to separate your data into training, test (holdout-response), and holdout (holdout-prompt) sets

set.seed(40)

set <- rep(NA, nrow(CPSTfulldataset2))

for (pid in unique(CPSTfulldataset2$ProblemID)) {
  # Get indices of rows with current ProblemID
  pid_idx <- which(CPSTfulldataset2$ProblemID == pid)
  
  # Randomly assign each row to training or test set
  num_samples <- length(pid_idx)
  train_size <- round(0.8 * num_samples)
  
  # Randomly permute the indices to assign randomly
  shuffled_idx <- sample(pid_idx)
  
  # Assign "training" to the first train_size samples
  set[shuffled_idx[1:train_size]] <- "training"
  
  # Assign "test" to the rest of the samples
  set[shuffled_idx[(train_size + 1):num_samples]] <- "test"
}


# loop through each unique value in ProblemID
for (pid in unique(CPSTfulldataset2$ProblemID)) {
  # get indices of rows with current ProblemID
  pid_idx <- which(CPSTfulldataset2$ProblemID == pid)
  
  # loop through each unique value in set for current ProblemID
  for (s in unique(CPSTfulldataset2$set[pid_idx])) {
    # get indices of rows with current set value
    s_idx <- which(CPSTfulldataset2$set[pid_idx] == s)
    
    # calculate number of rows and percentage of whole ProblemID size
    num_rows <- length(s_idx)
    perc_rows <- num_rows / length(pid_idx) * 100
    
    # print information about current set value
    cat(sprintf("ProblemID %s, set %s: %d rows (%.2f%%)\n", pid, s, num_rows, perc_rows))
  }
}

CPSTfulldataset2$set[CPSTfulldataset2$ProblemID == "Mike"] <- "heldout"


write_csv(CPSTfulldataset2, file = "CPSTfulldataset2.csv")
