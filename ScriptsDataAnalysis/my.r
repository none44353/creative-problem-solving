library(ggplot2) 
library(tidyverse)
library(janitor)

analyze_correlation_with_outlier_removal <- function(y_var, x_var, cutoff_val, plot_graph = FALSE, plot_title = "Title", corr_label_x = -2, corr_label_y = 2) {
  fit_model <- lm(y_var ~ x_var)
  cooks_dist <- cooks.distance(fit_model)
  above_cutoff_indices <- unique(which(cooks_dist > cutoff_val))

  if (length(above_cutoff_indices) > 0) {
    x_cleaned <- x_var[-above_cutoff_indices]
    y_cleaned <- y_var[-above_cutoff_indices]
  } else {
    x_cleaned <- x_var
    y_cleaned <- y_var
  }

  correlation_result <- cor.test(y_cleaned, x_cleaned, method = "pearson")
  corr_value <- correlation_result$estimate

  df_cleaned <- data.frame(label = y_cleaned, prediction = x_cleaned)
  df_cleaned$PredError <- df_cleaned$label - df_cleaned$prediction

  mean_value <- mean(df_cleaned$label)
  sd_value <- sd(df_cleaned$label)
  lower_threshold <- mean_value - 2 * sd_value
  upper_threshold <- mean_value + 2 * sd_value
  selected_values_low <- df_cleaned$prediction[df_cleaned$label < lower_threshold]
  selected_values_up <- df_cleaned$prediction[df_cleaned$label > upper_threshold]

  mean_prediction_extreme_labels <- mean(abs(c(selected_values_up, selected_values_low)))
  mean_prediction_mid_labels <- mean(abs(df_cleaned$prediction[
    df_cleaned$label >= lower_threshold & df_cleaned$label <= upper_threshold]))

  # 可选：绘制图形
  if (plot_graph) {
    # 绘制Cook's distance图
    plot(cooks.distance(fit_model), type = "b", pch = 18, col = "red",
         main = paste("Cook's Distance for", plot_title),
         xlab = "Observation Number", ylab = "Cook's Distance")
    abline(h = cutoff_val, lty = 2)

    # 绘制散点图
    p <- ggplot(df_cleaned, aes(x = prediction, y = label)) +
      geom_point() +
      geom_abline(intercept = 0, slope = 1, color = "black") + # y = x 对角线
      geom_smooth(method = "lm", color = "#ffa600", se = FALSE) + # 回归线
      theme_minimal() +
      xlab("Model Prediction") +
      ylab("Human Rating") +
      ggtitle(plot_title) + # 添加标题
      theme(
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        axis.text = element_text(size = 16, margin = margin(t = 20)),
        axis.title = element_text(size = 18, margin = margin(b = 20)),
        plot.title = element_text(size = 20, hjust = 0.5) # 标题居中
      ) +
      annotate("text", x = corr_label_x, y = corr_label_y,
               label = paste0("italic(r) == ", format(corr_value, digits = 2)),
               parse = TRUE, hjust = 0, vjust = 1, size = 6) +
      coord_cartesian(xlim = c(-2, 2), ylim = c(-2, 2))
    print(p)


    original_corr_value <- cor.test(y_var, x_var, method = "pearson")$estimate

    cat("---", plot_title, "---\n")
    cat("Original Pearson correlation:", original_corr_value, "\n")
    cat("Correlation after outlier removal:", corr_value, "\n")
    cat("Number of outliers removed:", length(above_cutoff_indices), "\n")
    cat("Mean absolute prediction for extreme labels:", mean_prediction_extreme_labels, "\n")
    cat("Mean absolute prediction for middle labels:", mean_prediction_mid_labels, "\n\n")
  }


  # 返回结果
  return(list(
    cleaned_correlation = corr_value,
    df_cleaned = df_cleaned
  ))
}


CPSTfulldataset2 <- read_csv("creative-problem-solving/Data/CPSTfulldataset2.csv") %>%
  clean_names() %>%
  mutate(across(where(is.character), as.factor)) %>%
  mutate(across(where(is.numeric), as.double))

print(names(CPSTfulldataset2))

cutofftotal = 4/ (nrow(CPSTfulldataset2) - 2)
labqualtotal <- c(CPSTfulldataset2$fac_scores_q)
laborigtotal <- c(CPSTfulldataset2$fac_scores_o)
wordcounttotal <- c(CPSTfulldataset2$wordcount)
DSItotal <- c(CPSTfulldataset2$dsi)

# DSI ~ originality
result_dsi_orig <- analyze_correlation_with_outlier_removal(laborigtotal, DSItotal, cutofftotal)
print(paste("DSI ~ originality (cleaned) Pearson correlation:", result_dsi_orig$cleaned_correlation))

# DSI ~ quality
result_dsi_qual <- analyze_correlation_with_outlier_removal(labqualtotal, DSItotal, cutofftotal)
print(paste("DSI ~ quality (cleaned) Pearson correlation:", result_dsi_qual$cleaned_correlation))

# DSI ~ wordcount
result_dsi_wc <- analyze_correlation_with_outlier_removal(wordcounttotal, DSItotal, cutofftotal)
print(paste("DSI ~ wordcount (cleaned) Pearson correlation:", result_dsi_wc$cleaned_correlation))

# originality ~ quality
result_orig_qual <- analyze_correlation_with_outlier_removal(laborigtotal, labqualtotal, cutofftotal)
print(paste("Originality ~ quality (cleaned) Pearson correlation:", result_orig_qual$cleaned_correlation))

# word count & originality
result_wc_orig <- analyze_correlation_with_outlier_removal(laborigtotal, wordcounttotal, cutofftotal)
print(paste("Word Count ~ originality (cleaned) Pearson correlation:", result_wc_orig$cleaned_correlation))

# word count & quality
result_wc_qual <- analyze_correlation_with_outlier_removal(labqualtotal, wordcounttotal, cutofftotal)
print(paste("Word Count ~ quality (cleaned) Pearson correlation:", result_wc_qual$cleaned_correlation))



AllModelPredCPST <- read_csv("creative-problem-solving/Data/AllModelPredCPST.csv") %>%
  clean_names() %>%
  mutate(across(where(is.character), as.factor)) %>%
  mutate(across(where(is.numeric), as.double))


#########PREDICTIONS & RATINGS
##
AllModelPredCPSTtest <- subset(AllModelPredCPST, set == "test")
AllModelPredCPSTheldout <- subset(AllModelPredCPST, set == "heldout")
##extract human ratings
labqualtest <- c(AllModelPredCPSTtest$label_quality)
labqualheldout <- c(AllModelPredCPSTheldout$label_quality)
laborigtest <- c(AllModelPredCPSTtest$label_originality)
laborigheldout <- c(AllModelPredCPSTheldout$label_originality)

dsitest <- c(AllModelPredCPSTtest$dsi)
dsiheldout <- c(AllModelPredCPSTheldout$dsi)
##cook's cutoff (4/N-k-1)
cutoffHeldout = 4/ (nrow(AllModelPredCPSTheldout) - 2)
cutoffTest = 4/ (nrow(AllModelPredCPSTtest) - 2)
##
######################

### RoBERTa quality test set
# predRoBERTa_qualtest <- c(AllModelPredCPSTtest$prediction_quality_ro_ber_ta)
# labqualtest_RoBERTa <- c(labqualtest)
# results_RoBERTa_test_quality <- analyze_correlation_with_outlier_removal(
#   y_var = labqualtest_RoBERTa,
#   x_var = predRoBERTa_qualtest,
#   cutoff_val = cutoffTest,
#   plot_graph = TRUE,
#   plot_title = "RoBERTa Test Set Quality Results",
# )

### RoBERTa quality held-out set
# predRoBERTa_qualheldout <- c(AllModelPredCPSTheldout$prediction_quality_ro_ber_ta)
# labqualheldout_RoBERTa <- c(labqualheldout)
# results_RoBERTa_heldout_quality <- analyze_correlation_with_outlier_removal(
#   y_var = labqualheldout_RoBERTa,
#   x_var = predRoBERTa_qualheldout,
#   cutoff_val = cutoffHeldout,
#   plot_graph = TRUE,
#   plot_title = "RoBERTa Held-out Set Quality Results",
# )

### RoBERTa originality test set
# predRoBERTa_origtest <- c(AllModelPredCPSTtest$prediction_originality_ro_ber_ta)
# laborigtest_RoBERTa <- c(laborigtest)
# results_RoBERTa_test_originality <- analyze_correlation_with_outlier_removal(
#   y_var = laborigtest_RoBERTa,
#   x_var = predRoBERTa_origtest,
#   cutoff_val = cutoffTest,
#   plot_graph = TRUE,
#   plot_title = "RoBERTa Test Set Originality Results",
# )

# ## RoBERTa originality held-out set
# predRoBERTa_origheldout <- c(AllModelPredCPSTheldout$prediction_originality_ro_ber_ta)
# laborigheldout_RoBERTa <- c(laborigheldout)
# results_RoBERTa_heldout_originality <- analyze_correlation_with_outlier_removal(
#   y_var = laborigheldout_RoBERTa,
#   x_var = predRoBERTa_origheldout,
#   cutoff_val = cutoffHeldout,
#   plot_graph = TRUE,
#   plot_title = "RoBERTa Held-out Set Originality Results",
# )

### GPT-2 quality test set
# predGPT_qualtest <- c(AllModelPredCPSTtest$prediction_quality_gpt2)
# labqualtest_GPT <- c(labqualtest)
# results_GPT2_test_quality <- analyze_correlation_with_outlier_removal(
#   y_var = labqualtest_GPT,
#   x_var = predGPT_qualtest,
#   cutoff_val = cutoffTest,
#   plot_graph = TRUE,
#   plot_title = "GPT-2 Test Set Quality Results",
# )

### GPT-2 quality held-out set
# predGPT_qualheldout <- c(AllModelPredCPSTheldout$prediction_quality_gpt2)
# labqualheldout_GPT <- c(labqualheldout)
# results_GPT2_heldout_quality <- analyze_correlation_with_outlier_removal(
#   y_var = labqualheldout_GPT,
#   x_var = predGPT_qualheldout,
#   cutoff_val = cutoffHeldout,
#   plot_graph = TRUE,
#   plot_title = "GPT-2 Held-out Set Quality Results",
# )

### GPT-2 originality test set
# predGPT_origtest <- c(AllModelPredCPSTtest$prediction_originality_gpt2)
# laborigtest_GPT <- c(laborigtest)
# results_GPT2_test_originality <- analyze_correlation_with_outlier_removal(
#     y_var = laborigtest_GPT,
#     x_var = predGPT_origtest,
#     cutoff_val = cutoffTest,
#     plot_graph = TRUE,
#     plot_title = "GPT-2 Test Set Originality Results",
# )

# ## GPT-2 originality held-out set
# predGPT_origheldout <- c(AllModelPredCPSTheldout$prediction_originality_gpt2)
# laborigheldout_GPT <- c(laborigheldout)
# results_GPT2_heldout_originality <- analyze_correlation_with_outlier_removal(
#     y_var = laborigheldout_GPT,
#     x_var = predGPT_origheldout,
#     cutoff_val = cutoffHeldout,
#     plot_graph = TRUE,
#     plot_title = "GPT-2 Held-out Set Originality Results",
# )


# # DSI ~ originality held-out set
# predDSI_origheldout <- c(AllModelPredCPSTheldout$dsi)
# laborigheldout_DSI <- c(laborigheldout)
# results_DSI_heldout_originality <- analyze_correlation_with_outlier_removal(
#   y_var = laborigheldout_DSI,
#   x_var = predDSI_origheldout,
#   cutoff_val = cutoffHeldout,
#   plot_graph = TRUE,
#   plot_title = "DSI Held-out Set Originality Results",
# )


OurResult <- read_csv("creative-problem-solving/Data/our_result.csv") %>%
  clean_names() %>%
  mutate(across(where(is.character), as.factor)) %>%
  mutate(across(where(is.numeric), as.double))
  
laborig <- c(OurResult$fac_scores)
pre <- c(OurResult$pred_6)
cutoff_ours <- 4 / (nrow(OurResult) - 2)
result_our <- analyze_correlation_with_outlier_removal(
  y_var = laborig,
  x_var = pre,
  cutoff_val = cutoff_ours,
  plot_graph = TRUE,
  plot_title = "Our Results",
  corr_label_x = -2, 
  corr_label_y = 2
)