######
library(naniar)
library(plotrix)
library(tidyverse)
library(docstring)
library(readxl)
library(data.table)
library("Hmisc")
library(lavaan)
library(semTools)
library(semPlot)
library("irr")
library(mice)
library(janitor)
library(psych)
library(dplyr)
library(ggplot2)


# 创建一个包含两个点的简单数据框
simple_data <- data.frame(
  x = c(1, 5),
  y = c(1, 5)
)

# 使用ggplot2画一条直线
ggplot(simple_data, aes(x = x, y = y)) +
  geom_line() # geom_line() 用来连接点形成线


CPSTfulldataset2 <- read_csv("creative-problem-solving/Data/CPSTfulldataset2.csv") %>%
  clean_names() %>%
  mutate(across(where(is.character), as.factor)) %>%
  mutate(across(where(is.numeric), as.double))

print(names(CPSTfulldataset2))

#####
#######DSI
labqualtotal <- c(CPSTfulldataset2$fac_scores_q)
laborigtotal <- c(CPSTfulldataset2$fac_scores_o)
wordcounttotal <- c(CPSTfulldataset2$wordcount)
DSItotal <- c(CPSTfulldataset2$dsi)

analyze_correlation_with_outlier_removal <- function(y_var, x_var, cutoff_val) {
  df_temp <- data.frame(y = y_var, x = x_var)
  fit_model <- lm(y ~ x, data = df_temp)
  cooks_dist <- cooks.distance(fit_model)
  above_cutoff_indices <- unique(which(cooks_dist > cutoff_val))

  if (length(above_cutoff_indices) > 0) {
    x_cleaned <- x_var[-above_cutoff_indices]
    y_cleaned <- y_var[-above_cutoff_indices]
  } else {
    x_cleaned <- x_var
    y_cleaned <- y_var
  }

  correlation_result <- cor(x_cleaned, y_cleaned, method = "pearson", use = "complete.obs")

  return(correlation_result)
}


cutofftotal = 4/ (nrow(CPSTfulldataset2) - 2)

# DSI ~ originality
cor_dsi_orig <- analyze_correlation_with_outlier_removal(laborigtotal, DSItotal, cutofftotal)
print(paste("DSI ~ originality (cleaned) Pearson correlation:", cor_dsi_orig))

# DSI ~ quality
cor_dsi_qual <- analyze_correlation_with_outlier_removal(labqualtotal, DSItotal, cutofftotal)
print(paste("DSI ~ quality (cleaned) Pearson correlation:", cor_dsi_qual))

# DSI ~ wordcount
cor_dsi_wc <- analyze_correlation_with_outlier_removal(wordcounttotal, DSItotal, cutofftotal)
print(paste("DSI ~ wordcount (cleaned) Pearson correlation:", cor_dsi_wc))

# originality ~ quality
cor_orig_qual <- analyze_correlation_with_outlier_removal(laborigtotal, labqualtotal, cutofftotal)
print(paste("Originality ~ quality (cleaned) Pearson correlation:", cor_orig_qual))

# word count & originality
cor_wc_orig <- analyze_correlation_with_outlier_removal(laborigtotal, wordcounttotal, cutofftotal)
print(paste("Word Count ~ originality (cleaned) Pearson correlation:", cor_wc_orig))

# word count & quality
cor_wc_qual <- analyze_correlation_with_outlier_removal(labqualtotal, wordcounttotal, cutofftotal)
print(paste("Word Count ~ quality (cleaned) Pearson correlation:", cor_wc_qual))


# ##############################################
# ##############################################
# ##############################################


AllModelPredCPST <- read_csv("creative-problem-solving/Data/AllModelPredCPST.csv") %>%
  clean_names() %>%
  mutate(across(where(is.character), as.factor)) %>%
  mutate(across(where(is.numeric), as.double))

print(names(AllModelPredCPST))

#########PREDICTIONS & RATINGS
##
AllModelPredCPSTtest <- subset(AllModelPredCPST, set == "test")
AllModelPredCPSTheldout <- subset(AllModelPredCPST, set == "heldout")
##extract human ratings
labqualtest <- c(AllModelPredCPSTtest$label_quality)
labqualheldout <- c(AllModelPredCPSTheldout$label_quality)
laborigtest <- c(AllModelPredCPSTtest$label_originality)
laborigheldout <- c(AllModelPredCPSTheldout$label_originality)
##cook's cutoff (4/N-k-1)
cutoffHeldout = 4/ (nrow(AllModelPredCPSTheldout) - 2)
cutoffTest = 4/ (nrow(AllModelPredCPSTtest) - 2)
##
######################

####
##RoBERTa quality
#test set
predRoBERTa_qualtest <- c(AllModelPredCPSTtest$prediction_quality_ro_ber_ta)
labqualtest_RoBERTa <- c(labqualtest)


corr_RoBERTa_qualtest <- analyze_correlation_with_outlier_removal(labqualtest_RoBERTa, predRoBERTa_qualtest, cutoffTest)
print(paste("Test Set: RoBERTa ~ quality (cleaned) Pearson correlation:", corr_RoBERTa_qualtest))


fitRoBERTatestqual = lm(labqualtest_RoBERTa ~ predRoBERTa_qualtest)
cooks_distRoBERTatestqual <- cooks.distance(fitRoBERTatestqual)
above_cutoff_indices_RoBERTatestqual <- unique(which(cooks_distRoBERTatestqual > cutoffTest))

length(above_cutoff_indices_RoBERTatestqual)

plot(cooks.distance(fitRoBERTatestqual),type="b",pch=18,col="red")
abline(h=cutoffTest,lty=2)

predRoBERTa_qualtest_cleaned <- predRoBERTa_qualtest[-above_cutoff_indices_RoBERTatestqual]
labqualtest_RoBERTa_cleaned <- labqualtest_RoBERTa[-above_cutoff_indices_RoBERTatestqual]

cor.test(labqualtest_RoBERTa_cleaned, predRoBERTa_qualtest_cleaned, method = "pearson")
corr_RoBERTa_qualtest <- cor(labqualtest_RoBERTa_cleaned, predRoBERTa_qualtest_cleaned, method = c("pearson"))

df_RoBERTa_qualtest <- data.frame(label = labqualtest_RoBERTa_cleaned,
  prediction = predRoBERTa_qualtest_cleaned)

ggplot(df_RoBERTa_qualtest, aes(x = prediction, y = label)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = "black") +
  geom_smooth(method = "lm", color = "#ffa600", se = FALSE) +
  theme_minimal() +
  xlab("Model Prediction") + ylab("Human Rating") +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        axis.text = element_text(size = 16, margin = margin(t = 20)),  
        axis.title = element_text(size = 18, margin = margin(b = 20))) +
  annotate("text", x = -2, y = 2, 
           label = expression(italic(r) == 0.83), 
           hjust = 0, vjust = 1, size = 6) + coord_cartesian(xlim = c(-2, 2), ylim = c(-2, 2))

df_RoBERTa_qualtest$PredErrorRobQualTest <- df_RoBERTa_qualtest$label - df_RoBERTa_qualtest$prediction

mean_value <- mean(df_RoBERTa_qualtest$label)
sd_value <- sd(df_RoBERTa_qualtest$label)

lower_threshold <- mean_value - 2 * sd_value
selected_values_low <- df_RoBERTa_qualtest$prediction[df_RoBERTa_qualtest$label < lower_threshold]
upper_threshold <- mean_value + 2 * sd_value
selected_values_up <- df_RoBERTa_qualtest$prediction[df_RoBERTa_qualtest$label > upper_threshold]

selected_uplow <- abs(c(selected_values_up, selected_values_low))
mean(selected_uplow)
selected_values_between <- abs(df_RoBERTa_qualtest$prediction[
  df_RoBERTa_qualtest$label >= lower_threshold & df_RoBERTa_qualtest$label <= upper_threshold])
mean(selected_values_between)

label = expression(italic(r) == 0.81)

#heldout set
predRoBERTa_qualheldout <- c(AllModelPredCPSTheldout$predictionQuality_RoBERTa)
labqualheldout_RoBERTa <- c(labqualheldout)

fitRoBERTaheldoutqual = lm(labqualheldout_RoBERTa ~ predRoBERTa_qualheldout)
cooks_distRoBERTaheldoutqual <- cooks.distance(fitRoBERTaheldoutqual)
above_cutoff_indices_RoBERTaheldoutqual <- unique(which(cooks_distRoBERTaheldoutqual > cutoffHeldout))

length(above_cutoff_indices_RoBERTaheldoutqual)

plot(cooks.distance(fitRoBERTaheldoutqual),type="b",pch=18,col="red")
abline(h=cutoffHeldout,lty=2)

predRoBERTa_qualheldout_cleaned <- predRoBERTa_qualheldout[-above_cutoff_indices_RoBERTaheldoutqual]
labqualheldout_RoBERTa_cleaned <- labqualheldout_RoBERTa[-above_cutoff_indices_RoBERTaheldoutqual]

cor.test(labqualheldout_RoBERTa_cleaned, predRoBERTa_qualheldout_cleaned, method = "pearson")
corr_RoBERTa_qualheldout <- cor(labqualheldout_RoBERTa_cleaned, predRoBERTa_qualheldout_cleaned, method = c("pearson"))

df_RoBERTa_qualheldout <- data.frame(label = labqualheldout_RoBERTa_cleaned,
                                  prediction = predRoBERTa_qualheldout_cleaned)

ggplot(df_RoBERTa_qualheldout, aes(x = prediction, y = label)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = "black") +
  geom_smooth(method = "lm", color = "orange", se = FALSE) +
  theme_minimal() +
  xlab("Model Prediction") + ylab("Human Rating") +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        axis.text = element_text(size = 16, margin = margin(t = 20)),  
        axis.title = element_text(size = 18, margin = margin(b = 20))) +
  annotate("text", x = -2, y = 2, 
           label = expression(italic(r) == 0.89), 
           hjust = 0, vjust = 1, size = 6) + coord_cartesian(xlim = c(-2, 2), ylim = c(-2, 2))

df_RoBERTa_qualheldout$PredErrorRobQualTest <- df_RoBERTa_qualheldout$label - df_RoBERTa_qualheldout$prediction

mean_value <- mean(df_RoBERTa_qualheldout$label)
sd_value <- sd(df_RoBERTa_qualheldout$label)

lower_threshold <- mean_value - 2 * sd_value
selected_values_low <- df_RoBERTa_qualheldout$prediction[df_RoBERTa_qualheldout$label < lower_threshold]
upper_threshold <- mean_value + 2 * sd_value
selected_values_up <- df_RoBERTa_qualheldout$prediction[df_RoBERTa_qualheldout$label > upper_threshold]

selected_uplow <- abs(c(selected_values_up, selected_values_low))
mean(selected_uplow)
selected_values_between <- abs(df_RoBERTa_qualheldout$prediction[
  df_RoBERTa_qualheldout$label >= lower_threshold & df_RoBERTa_qualheldout$label <= upper_threshold])
mean(selected_values_between)



##RoBERTa originality
#test set
predRoBERTa_origtest <- c(AllModelPredCPSTtest$predictionOriginality_RoBERTa)
laborigtest_RoBERTa <- c(laborigtest)

fitRoBERTatestorig = lm(laborigtest_RoBERTa ~ predRoBERTa_origtest)
cooks_distRoBERTatestorig <- cooks.distance(fitRoBERTatestorig)
above_cutoff_indices_RoBERTatestorig <- unique(which(cooks_distRoBERTatestorig > cutoffTest))

length(above_cutoff_indices_RoBERTatestorig)

plot(cooks.distance(fitRoBERTatestorig),type="b",pch=18,col="red")
abline(h=cutoffTest,lty=2)

predRoBERTa_origtest_cleaned <- predRoBERTa_origtest[-above_cutoff_indices_RoBERTatestorig]
laborigtest_RoBERTa_cleaned <- laborigtest_RoBERTa[-above_cutoff_indices_RoBERTatestorig]

cor.test(laborigtest_RoBERTa_cleaned, predRoBERTa_origtest_cleaned, method = "pearson")
corr_RoBERTa_origtest <- cor(laborigtest_RoBERTa_cleaned, predRoBERTa_origtest_cleaned, method = c("pearson"))

df_RoBERTa_origtest_cleaned <- data.frame(label = laborigtest_RoBERTa_cleaned,
                                     prediction = predRoBERTa_origtest_cleaned)

ggplot(df_RoBERTa_origtest_cleaned, aes(x = prediction, y = label)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = "black") +
  geom_smooth(method = "lm", color = "orange", se = FALSE) +
  theme_minimal() +
  xlab("Model Prediction") + ylab("Human Rating") +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        axis.text = element_text(size = 16, margin = margin(t = 20)),  
        axis.title = element_text(size = 18, margin = margin(b = 20))) +
  annotate("text", x = -2, y = 2, 
           label = expression(italic(r) == 0.79), 
           hjust = 0, vjust = 1, size = 6) + coord_cartesian(xlim = c(-2, 2), ylim = c(-2, 2))

df_RoBERTa_origtest_cleaned$PredErrorRobQualTest <- 
  df_RoBERTa_origtest_cleaned$label - df_RoBERTa_origtest_cleaned$prediction

mean_value <- mean(df_RoBERTa_origtest_cleaned$label)
sd_value <- sd(df_RoBERTa_origtest_cleaned$label)

lower_threshold <- mean_value - 2 * sd_value
selected_values_low <- df_RoBERTa_origtest_cleaned$prediction[df_RoBERTa_origtest_cleaned$label < lower_threshold]
upper_threshold <- mean_value + 2 * sd_value
selected_values_up <- df_RoBERTa_origtest_cleaned$prediction[df_RoBERTa_origtest_cleaned$label > upper_threshold]

selected_uplow <- abs(c(selected_values_up, selected_values_low))
mean(selected_uplow)
selected_values_between <- abs(df_RoBERTa_origtest_cleaned$prediction[
  df_RoBERTa_origtest_cleaned$label >= lower_threshold & df_RoBERTa_origtest_cleaned$label <= upper_threshold])
mean(selected_values_between)

#heldout set
predRoBERTa_origheldout <- c(AllModelPredCPSTheldout$predictionOriginality_RoBERTa)
laborigheldout_RoBERTa <- c(laborigheldout)

fitRoBERTaheldoutorig = lm(laborigheldout_RoBERTa ~ predRoBERTa_origheldout)
cooks_distRoBERTaheldoutorig <- cooks.distance(fitRoBERTaheldoutorig)
above_cutoff_indices_RoBERTaheldoutorig <- unique(which(cooks_distRoBERTaheldoutorig > cutoffHeldout))

length(above_cutoff_indices_RoBERTaheldoutorig)

plot(cooks.distance(fitRoBERTaheldoutorig),type="b",pch=18,col="red")
abline(h=cutoffHeldout,lty=2)

predRoBERTa_origheldout_cleaned <- predRoBERTa_origheldout[-above_cutoff_indices_RoBERTaheldoutorig]
laborigheldout_RoBERTa_cleaned <- laborigheldout_RoBERTa[-above_cutoff_indices_RoBERTaheldoutorig]

cor.test(laborigheldout_RoBERTa_cleaned, predRoBERTa_origheldout_cleaned, method = "pearson")
corr_RoBERTa_origheldout <- cor(laborigheldout_RoBERTa_cleaned, predRoBERTa_origheldout_cleaned, method = c("pearson"))

df_RoBERTa_origheldout_cleaned <- data.frame(label = laborigheldout_RoBERTa_cleaned,
                                          prediction = predRoBERTa_origheldout_cleaned)

ggplot(df_RoBERTa_origheldout_cleaned, aes(x = prediction, y = label)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = "black") +
  geom_smooth(method = "lm", color = "orange", se = FALSE) +
  theme_minimal() +
  xlab("Model Prediction") + ylab("Human Rating") +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        axis.text = element_text(size = 16, margin = margin(t = 20)),  
        axis.title = element_text(size = 18, margin = margin(b = 20))) +
  annotate("text", x = -2, y = 2, 
           label = expression(italic(r) == 0.41), 
           hjust = 0, vjust = 1, size = 6) + coord_cartesian(xlim = c(-2, 2), ylim = c(-2, 2))


df_RoBERTa_origheldout_cleaned$PredErrorRobQualTest <- 
  df_RoBERTa_origheldout_cleaned$label - df_RoBERTa_origheldout_cleaned$prediction


mean_value <- mean(df_RoBERTa_origheldout_cleaned$label)
sd_value <- sd(df_RoBERTa_origheldout_cleaned$label)

lower_threshold <- mean_value - 2 * sd_value
selected_values_low <- df_RoBERTa_origheldout_cleaned$prediction[df_RoBERTa_origheldout_cleaned$label < lower_threshold]
upper_threshold <- mean_value + 2 * sd_value
selected_values_up <- df_RoBERTa_origheldout_cleaned$prediction[df_RoBERTa_origheldout_cleaned$label > upper_threshold]

selected_uplow <- abs(c(selected_values_up, selected_values_low))
mean(selected_uplow)
selected_values_between <- abs(df_RoBERTa_origheldout_cleaned$prediction[
  df_RoBERTa_origheldout_cleaned$label >= lower_threshold & df_RoBERTa_origheldout_cleaned$label <= upper_threshold])
mean(selected_values_between)

# #####
# #######
# #####


##GPT-2 quality
#test set
predGPT_qualtest <- c(AllModelPredCPSTtest$predictionQuality_GPT2)
labqualtest_GPT <- c(labqualtest)

fitGPTtestqual = lm(labqualtest_GPT ~ predGPT_qualtest)
cooks_distGPTtestqual <- cooks.distance(fitGPTtestqual)
above_cutoff_indices_GPTtestqual <- unique(which(cooks_distGPTtestqual > cutoffTest))

length(above_cutoff_indices_GPTtestqual)

plot(cooks.distance(fitGPTtestqual),type="b",pch=18,col="red")
abline(h=cutoffTest,lty=2)

predGPT_qualtest_cleaned <- predGPT_qualtest[-above_cutoff_indices_GPTtestqual]
labqualtest_GPT_cleaned <- labqualtest_GPT[-above_cutoff_indices_GPTtestqual]

cor.test(labqualtest_GPT_cleaned, predGPT_qualtest_cleaned, method = "pearson")
corr_GPT_qualtest <- cor(labqualtest_GPT_cleaned, predGPT_qualtest_cleaned, method = c("pearson"))

df_GPT_qualtest_cleaned <- data.frame(label = labqualtest_GPT_cleaned,
                                          prediction = predGPT_qualtest_cleaned)

ggplot(df_GPT_qualtest_cleaned, aes(x = prediction, y = label)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = "black") +
  geom_smooth(method = "lm", color = "orange", se = FALSE) +
  theme_minimal() +
  xlab("Model Prediction") + ylab("Human Rating") +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        axis.text = element_text(size = 16, margin = margin(t = 20)),  
        axis.title = element_text(size = 18, margin = margin(b = 20))) +
  annotate("text", x = -2, y = 2, 
           label = expression(italic(r) == 0.83), 
           hjust = 0, vjust = 1, size = 6) + coord_cartesian(xlim = c(-2, 2), ylim = c(-2, 2))


df_GPT_qualtest_cleaned$PredErrorRobQualTest <- 
  df_GPT_qualtest_cleaned$label - df_GPT_qualtest_cleaned$prediction

mean_value <- mean(df_GPT_qualtest_cleaned$label)
sd_value <- sd(df_GPT_qualtest_cleaned$label)

lower_threshold <- mean_value - 2 * sd_value
selected_values_low <- df_GPT_qualtest_cleaned$prediction[df_GPT_qualtest_cleaned$label < lower_threshold]
upper_threshold <- mean_value + 2 * sd_value
selected_values_up <- df_GPT_qualtest_cleaned$prediction[df_GPT_qualtest_cleaned$label > upper_threshold]

selected_uplow <- abs(c(selected_values_up, selected_values_low))
mean(selected_uplow)
selected_values_between <- abs(df_GPT_qualtest_cleaned$prediction[
  df_GPT_qualtest_cleaned$label >= lower_threshold & df_GPT_qualtest_cleaned$label <= upper_threshold])
mean(selected_values_between)

# #heldout set
# predGPT_qualheldout <- c(AllModelPredCPSTheldout$predictionQuality_GPT2)
# labqualheldout_GPT <- c(labqualheldout)

# fitGPTheldoutqual = lm(labqualheldout_GPT ~ predGPT_qualheldout)
# cooks_distGPTheldoutqual <- cooks.distance(fitGPTheldoutqual)
# above_cutoff_indices_GPTheldoutqual <- unique(which(cooks_distGPTheldoutqual > cutoffHeldout))

# length(above_cutoff_indices_GPTheldoutqual)

# plot(cooks.distance(fitGPTheldoutqual),type="b",pch=18,col="red")
# abline(h=cutoffHeldout,lty=2)

# predGPT_qualheldout_cleaned <- predGPT_qualheldout[-above_cutoff_indices_GPTheldoutqual]
# labqualheldout_GPT_cleaned <- labqualheldout_GPT[-above_cutoff_indices_GPTheldoutqual]

# cor.test(labqualheldout_GPT_cleaned, predGPT_qualheldout_cleaned, method = "pearson")
# corr_GPT_qualheldout <- cor(labqualheldout_GPT_cleaned, predGPT_qualheldout_cleaned, method = c("pearson"))

# df_GPT_qualheldout_cleaned <- data.frame(label = labqualheldout_GPT_cleaned,
#                                       prediction = predGPT_qualheldout_cleaned)

# ggplot(df_GPT_qualheldout_cleaned, aes(x = prediction, y = label)) +
#   geom_point() +
#   geom_abline(intercept = 0, slope = 1, color = "black") +
#   geom_smooth(method = "lm", color = "orange", se = FALSE) +
#   theme_minimal() +
#   xlab("Model Prediction") + ylab("Human Rating") +
#   theme(panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),
#         panel.background = element_blank(),
#         axis.line = element_line(colour = "black"),
#         axis.text = element_text(size = 16, margin = margin(t = 20)),  
#         axis.title = element_text(size = 18, margin = margin(b = 20))) +
#   annotate("text", x = -2, y = 2, 
#            label = expression(italic(r) == 0.89), 
#            hjust = 0, vjust = 1, size = 6) + coord_cartesian(xlim = c(-2, 2), ylim = c(-2, 2))


# df_GPT_qualheldout_cleaned$PredErrorRobQualTest <- 
#   df_GPT_qualheldout_cleaned$label - df_GPT_qualheldout_cleaned$prediction

# mean_value <- mean(df_GPT_qualheldout_cleaned$label)
# sd_value <- sd(df_GPT_qualheldout_cleaned$label)

# lower_threshold <- mean_value - 2 * sd_value
# selected_values_low <- df_GPT_qualheldout_cleaned$prediction[df_GPT_qualheldout_cleaned$label < lower_threshold]
# upper_threshold <- mean_value + 2 * sd_value
# selected_values_up <- df_GPT_qualheldout_cleaned$prediction[df_GPT_qualheldout_cleaned$label > upper_threshold]

# selected_uplow <- abs(c(selected_values_up, selected_values_low))
# mean(selected_uplow)
# selected_values_between <- abs(df_GPT_qualheldout_cleaned$prediction[
#   df_GPT_qualheldout_cleaned$label >= lower_threshold & df_GPT_qualheldout_cleaned$label <= upper_threshold])
# mean(selected_values_between)

# ##GPT-2 originality
# #test set
# predGPT_origtest <- c(AllModelPredCPSTtest$predictionOriginality_GPT2)
# laborigtest_GPT <- c(laborigtest)

# fitGPTtestorig = lm(laborigtest_GPT ~ predGPT_origtest)
# cooks_distGPTtestorig <- cooks.distance(fitGPTtestorig)
# above_cutoff_indices_GPTtestorig <- unique(which(cooks_distGPTtestorig > cutoffTest))

# length(above_cutoff_indices_GPTtestorig)

# plot(cooks.distance(fitGPTtestorig),type="b",pch=18,col="red")
# abline(h=cutoffTest,lty=2)

# predGPT_origtest_cleaned <- predGPT_origtest[-above_cutoff_indices_GPTtestorig]
# laborigtest_GPT_cleaned <- laborigtest_GPT[-above_cutoff_indices_GPTtestorig]

# cor.test(laborigtest_GPT_cleaned, predGPT_origtest_cleaned, method = "pearson")
# corr_GPT_origtest <- cor(laborigtest_GPT_cleaned, predGPT_origtest_cleaned, method = c("pearson"))

# df_GPT_origtest_cleaned <- data.frame(label = laborigtest_GPT_cleaned,
#                                          prediction = predGPT_origtest_cleaned)

# ggplot(df_GPT_origtest_cleaned, aes(x = prediction, y = label)) +
#   geom_point() +
#   geom_abline(intercept = 0, slope = 1, color = "black") +
#   geom_smooth(method = "lm", color = "orange", se = FALSE) +
#   theme_minimal() +
#   xlab("Model Prediction") + ylab("Human Rating") +
#   theme(panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),
#         panel.background = element_blank(),
#         axis.line = element_line(colour = "black"),
#         axis.text = element_text(size = 16, margin = margin(t = 20)),  
#         axis.title = element_text(size = 18, margin = margin(b = 20))) +
#   annotate("text", x = -2, y = 2, 
#            label = expression(italic(r) == "0.80"), 
#            hjust = 0, vjust = 1, size = 6) + coord_cartesian(xlim = c(-2, 2), ylim = c(-2, 2))

# df_GPT_origtest_cleaned$PredErrorRobQualTest <- 
#   df_GPT_origtest_cleaned$label - df_GPT_origtest_cleaned$prediction

# mean_value <- mean(df_GPT_origtest_cleaned$label)
# sd_value <- sd(df_GPT_origtest_cleaned$label)

# lower_threshold <- mean_value - 2 * sd_value
# selected_values_low <- df_GPT_origtest_cleaned$prediction[df_GPT_origtest_cleaned$label < lower_threshold]
# upper_threshold <- mean_value + 2 * sd_value
# selected_values_up <- df_GPT_origtest_cleaned$prediction[df_GPT_origtest_cleaned$label > upper_threshold]

# selected_uplow <- abs(c(selected_values_up, selected_values_low))
# mean(selected_uplow)
# selected_values_between <- abs(df_GPT_origtest_cleaned$prediction[
#   df_GPT_origtest_cleaned$label >= lower_threshold & df_GPT_origtest_cleaned$label <= upper_threshold])
# mean(selected_values_between)

# #heldout set
# predGPT_origheldout <- c(AllModelPredCPSTheldout$predictionOriginality_GPT2)
# laborigheldout_GPT <- c(laborigheldout)

# fitGPTheldoutorig = lm(laborigheldout_GPT ~ predGPT_origheldout)
# cooks_distGPTheldoutorig <- cooks.distance(fitGPTheldoutorig)
# above_cutoff_indices_GPTheldoutorig <- unique(which(cooks_distGPTheldoutorig > cutoffHeldout))

# length(above_cutoff_indices_GPTheldoutorig)

# plot(cooks.distance(fitGPTheldoutorig),type="b",pch=18,col="red")
# abline(h=cutoffHeldout,lty=2)

# predGPT_origheldout_cleaned <- predGPT_origheldout[-above_cutoff_indices_GPTheldoutorig]
# laborigheldout_GPT_cleaned <- laborigheldout_GPT[-above_cutoff_indices_GPTheldoutorig]

# cor.test(laborigheldout_GPT_cleaned, predGPT_origheldout_cleaned, method = "pearson")
# corr_GPT_origheldout <- cor(laborigheldout_GPT_cleaned, predGPT_origheldout_cleaned, method = c("pearson"))

# df_GPT_origheldout_cleaned <- data.frame(label = laborigheldout_GPT_cleaned,
#                                       prediction = predGPT_origheldout_cleaned)

# ggplot(df_GPT_origheldout_cleaned, aes(x = prediction, y = label)) +
#   geom_point() +
#   geom_abline(intercept = 0, slope = 1, color = "black") +
#   geom_smooth(method = "lm", color = "orange", se = FALSE) +
#   theme_minimal() +
#   xlab("Model Prediction") + ylab("Human Rating") +
#   theme(panel.grid.major = element_blank(),
#         panel.grid.minor = element_blank(),
#         panel.background = element_blank(),
#         axis.line = element_line(colour = "black"),
#         axis.text = element_text(size = 16, margin = margin(t = 20)),  
#         axis.title = element_text(size = 18, margin = margin(b = 20))) +
#         annotate("text", x = -2, y = 2, 
#            label = expression(italic(r) == "0.40"), 
#            hjust = 0, vjust = 1, size = 6) + coord_cartesian(xlim = c(-2, 2), ylim = c(-2, 2))

# df_GPT_origheldout_cleaned$PredErrorRobQualTest <- 
#   df_GPT_origheldout_cleaned$label - df_GPT_origheldout_cleaned$prediction

# mean_value <- mean(df_GPT_origheldout_cleaned$label)
# sd_value <- sd(df_GPT_origheldout_cleaned$label)

# lower_threshold <- mean_value - 2 * sd_value
# selected_values_low <- df_GPT_origheldout_cleaned$prediction[df_GPT_origheldout_cleaned$label < lower_threshold]
# upper_threshold <- mean_value + 2 * sd_value
# selected_values_up <- df_GPT_origheldout_cleaned$prediction[df_GPT_origheldout_cleaned$label > upper_threshold]

# selected_uplow <- abs(c(selected_values_up, selected_values_low))
# mean(selected_uplow)
# selected_values_between <- abs(df_GPT_origheldout_cleaned$prediction[
#   df_GPT_origheldout_cleaned$label >= lower_threshold & df_GPT_origheldout_cleaned$label <= upper_threshold])
# mean(selected_values_between)

