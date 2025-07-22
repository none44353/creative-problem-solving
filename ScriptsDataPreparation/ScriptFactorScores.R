#Run this script to take the individual datasets from each study, calculate factor scores, and combine them into a single dataset

######
library(naniar)
library(plotrix)
library(tidyverse)
library(docstring)
library(readxl)
library(data.table)
library(psych)
library(dplyr)
library(janitor)
library("Hmisc")
library(lavaan)
library(semTools)
library(semPlot)
library("irr")
library(mice)


#download first DF
#originality
CPST1 <- read_csv("Becky1.csv")

CPST1 <- data.frame(lapply(CPST1, function(x) ifelse(x == "#NULL!", NA, x)))
CPST1 <- na.omit(CPST1)

CPST1$Orig_1 <- as.numeric(CPST1$Orig_1)
CPST1$Orig_2 <- as.numeric(CPST1$Orig_2)
CPST1$Orig_3 <- as.numeric(CPST1$Orig_3)


cfa_model1 <-
  '
  LatentO =~
  Orig_1 +
  Orig_2 +
  Orig_3
 
   LatentO ~~ 1*LatentO
  '
SEM1 <- sem(cfa_model1, data = CPST1)
fac.scores1 <- lavPredict(SEM1)
CPST1$FacScoresO <- c(fac.scores1)

CPST1$MeanO <- (CPST1$Orig_1 + CPST1$Orig_2 + CPST1$Orig_3) / 3

summary(SEM1, fit.measures = TRUE)
fitMeasures(SEM1, c("chisq", "df", "pvalue"))

#quality
CPST1$Qual_4 <- as.numeric(CPST1$Qual_4)
CPST1$Qual_5 <- as.numeric(CPST1$Qual_5)


cfa_model2 <-
  '
  LatentQ =~
  Qual_4 +
  Qual_5 
   LatentQ ~~ 1*LatentQ
  '

SEM2 <- sem(cfa_model2, data = CPST1)
fac.scores2 <- lavPredict(SEM2)
CPST1$FacScoresQ <- c(fac.scores2)

CPST1$MeanQ <- (CPST1$Qual_4 + CPST1$Qual_5) / 2

summary(SEM2, fit.measures = TRUE)
fitMeasures(SEM2, c("chisq", "df", "pvalue"))

CPST1$Dataset <- "CPST1"

CPST1 <- dplyr::select(CPST1, "Problem", "Solutions", "FacScoresQ", "FacScoresO", "MeanQ", "MeanO", "Dataset")



######################
############
#download second DF
#originality
CPST2 <- read_csv("ACME1.csv")

CPST2 <- data.frame(lapply(CPST2, function(x) ifelse(x == "#NULL!", NA, x)))
CPST2 <- na.omit(CPST2)

CPST2$Orig_1 <- as.numeric(CPST2$Orig_1)
CPST2$Orig_2 <- as.numeric(CPST2$Orig_2)
CPST2$Orig_3 <- as.numeric(CPST2$Orig_3)
CPST2$Orig_4 <- as.numeric(CPST2$Orig_4)

cfa_model3 <-
  '
  LatentO =~
  Orig_1 +
  Orig_2 +
  Orig_3 +
  Orig_4
 
   LatentO ~~ 1*LatentO
  '
SEM3 <- sem(cfa_model3, data = CPST2)
fac.scores3 <- lavPredict(SEM3)
CPST2$FacScoresO <- c(fac.scores3)

CPST2$MeanO <- (CPST2$Orig_1 + CPST2$Orig_2 + CPST2$Orig_3 + CPST2$Orig_4) / 4

summary(SEM3, fit.measures = TRUE)
fitMeasures(SEM3, c("chisq", "df", "pvalue"))

#quality
CPST2$Qual_1 <- as.numeric(CPST2$Qual_1)
CPST2$Qual_2 <- as.numeric(CPST2$Qual_2)
CPST2$Qual_3 <- as.numeric(CPST2$Qual_3)
CPST2$Qual_4 <- as.numeric(CPST2$Qual_4)

cfa_model4 <-
  '
  LatentQ =~
  Qual_1 +
  Qual_2 +
  Qual_3 +
  Qual_4
 
   LatentQ ~~ 1*LatentQ
  '

SEM4 <- sem(cfa_model4, data = CPST2)
fac.scores4 <- lavPredict(SEM4)
CPST2$FacScoresQ <- c(fac.scores4)

CPST2$MeanQ <- (CPST2$Qual_1 + CPST2$Qual_2 + CPST2$Qual_3 + CPST2$Qual_4) / 4

summary(SEM4, fit.measures = TRUE)
fitMeasures(SEM4, c("chisq", "df", "pvalue"))

CPST2$Dataset <- "CPST2"

CPST2 <- dplyr::select(CPST2, "Problem", "Solutions", "FacScoresQ", "FacScoresO", "MeanQ", "MeanO", "Dataset")



######################
############
#download third DF
#originality
CPST3 <- read_csv("Clara1.csv")

CPST3 <- data.frame(lapply(CPST3, function(x) ifelse(x == "#NULL!", NA, x)))
CPST3 <- na.omit(CPST3)

CPST3$Orig_1 <- as.numeric(CPST3$Orig_1)
CPST3$Orig_2 <- as.numeric(CPST3$Orig_2)
CPST3$Orig_3 <- as.numeric(CPST3$Orig_3)


cfa_model5 <-
  '
  LatentO =~
  Orig_1 +
  Orig_2 +
  Orig_3
 
   LatentO ~~ 1*LatentO
  '
SEM5 <- sem(cfa_model5, data = CPST3)
fac.scores5 <- lavPredict(SEM5)
CPST3$FacScoresO <- c(fac.scores5)

CPST3$MeanO <- (CPST3$Orig_1 + CPST3$Orig_2 + CPST3$Orig_3) / 3

summary(SEM5, fit.measures = TRUE)
fitMeasures(SEM5, c("chisq", "df", "pvalue"))

#quality
CPST3$Qual_1 <- as.numeric(CPST3$Qual_1)
CPST3$Qual_2 <- as.numeric(CPST3$Qual_2)
CPST3$Qual_3 <- as.numeric(CPST3$Qual_3)


cfa_model6 <-
  '
  LatentQ =~
  Qual_1 +
  Qual_2 +
  Qual_3
 
   LatentQ ~~ 1*LatentQ
  '

SEM6 <- sem(cfa_model6, data = CPST3)
fac.scores6 <- lavPredict(SEM6)
CPST3$FacScoresQ <- c(fac.scores6)

CPST3$MeanQ <- (CPST3$Qual_1 + CPST3$Qual_2 + CPST3$Qual_3) / 3

summary(SEM6, fit.measures = TRUE)
fitMeasures(SEM6, c("chisq", "df", "pvalue"))

CPST3$Dataset <- "CPST3"

CPST3 <- dplyr::select(CPST3, "Problem", "Solutions", "FacScoresQ", "FacScoresO", "MeanQ", "MeanO", "Dataset")



######################
############
#download four DF
#originality
CPST4 <- read_csv("ACME2.csv")

CPST4 <- data.frame(lapply(CPST4, function(x) ifelse(x == "#NULL!", NA, x)))
CPST4 <- na.omit(CPST4)

CPST4$Orig_1 <- as.numeric(CPST4$Orig_1)
CPST4$Orig_2 <- as.numeric(CPST4$Orig_2)
CPST4$Orig_3 <- as.numeric(CPST4$Orig_3)


cfa_model7 <-
  '
  LatentO =~
  Orig_1 +
  Orig_2 +
  Orig_3
 
   LatentO ~~ 1*LatentO
  '
SEM7 <- sem(cfa_model7, data = CPST4)
fac.scores7 <- lavPredict(SEM7)
CPST4$FacScoresO <- c(fac.scores7)

CPST4$MeanO <- (CPST4$Orig_1 + CPST4$Orig_2 + CPST4$Orig_3) / 3

summary(SEM7, fit.measures = TRUE)
fitMeasures(SEM7, c("chisq", "df", "pvalue"))

#quality
CPST4$Qual_1 <- as.numeric(CPST4$Qual_1)
CPST4$Qual_2 <- as.numeric(CPST4$Qual_2)
CPST4$Qual_3 <- as.numeric(CPST4$Qual_3)


cfa_model8 <-
  '
  LatentQ =~
  Qual_1 +
  Qual_2 +
  Qual_3
 
   LatentQ ~~ 1*LatentQ
  '

SEM8 <- sem(cfa_model8, data = CPST4)
fac.scores8 <- lavPredict(SEM8)
CPST4$FacScoresQ <- c(fac.scores8)

CPST4$MeanQ <- (CPST4$Qual_1 + CPST4$Qual_2 + CPST4$Qual_3) / 3

summary(SEM8, fit.measures = TRUE)
fitMeasures(SEM8, c("chisq", "df", "pvalue"))

CPST4$Dataset <- "CPST4"

CPST4 <- dplyr::select(CPST4, "Problem", "Solutions", "FacScoresQ", "FacScoresO", "MeanQ", "MeanO", "Dataset")



######################
############
#download fifth A DF
#originality
CPST5 <- read_csv("Becky2.csv")

CPST5 <- data.frame(lapply(CPST5, function(x) ifelse(x == "#NULL!", NA, x)))
CPST5 <- na.omit(CPST5)

CPST5$Orig_1 <- as.numeric(CPST5$Orig_1)
CPST5$Orig_2 <- as.numeric(CPST5$Orig_2)
CPST5$Orig_3 <- as.numeric(CPST5$Orig_3)


cfa_model9 <-
  '
  LatentO =~
  Orig_1 +
  Orig_2 +
  Orig_3
 
   LatentO ~~ 1*LatentO
  '
SEM9 <- sem(cfa_model9, data = CPST5)
fac.scores9 <- lavPredict(SEM9)
CPST5$FacScoresO <- c(fac.scores9)

CPST5$MeanO <- (CPST5$Orig_1 + CPST5$Orig_2 + CPST5$Orig_3) / 3

summary(SEM9, fit.measures = TRUE)
fitMeasures(SEM9, c("chisq", "df", "pvalue"))

#quality
CPST5$Qual_1 <- as.numeric(CPST5$Qual_1)
CPST5$Qual_2 <- as.numeric(CPST5$Qual_2)
CPST5$Qual_3 <- as.numeric(CPST5$Qual_3)


cfa_model10 <-
  '
  LatentQ =~
  Qual_1 +
  Qual_2 +
  Qual_3
 
   LatentQ ~~ 1*LatentQ
  '

SEM10 <- sem(cfa_model10, data = CPST5)
fac.scores10 <- lavPredict(SEM10)
CPST5$FacScoresQ <- c(fac.scores10)

CPST5$MeanQ <- (CPST5$Qual_1 + CPST5$Qual_2 + CPST5$Qual_3) / 3

summary(SEM10, fit.measures = TRUE)
fitMeasures(SEM10, c("chisq", "df", "pvalue"))

CPST5$Dataset <- "CPST5"

CPST5 <- dplyr::select(CPST5, "Problem", "Solutions", "FacScoresQ", "FacScoresO", "MeanQ", "MeanO", "Dataset")



######################
############
#download sixth A DF
#originality
CPST6 <- read_csv("Mike1.csv")

CPST6 <- data.frame(lapply(CPST6, function(x) ifelse(x == "#NULL!", NA, x)))
CPST6 <- na.omit(CPST6)

CPST6$Orig_1 <- as.numeric(CPST6$Orig_1)
CPST6$Orig_2 <- as.numeric(CPST6$Orig_2)
CPST6$Orig_3 <- as.numeric(CPST6$Orig_3)


cfa_model11 <-
  '
  LatentO =~
  Orig_1 +
  Orig_2 +
  Orig_3
 
   LatentO ~~ 1*LatentO
  '
SEM11 <- sem(cfa_model11, data = CPST6)
fac.scores11 <- lavPredict(SEM11)
CPST6$FacScoresO <- c(fac.scores11)

CPST6$MeanO <- (CPST6$Orig_1 + CPST6$Orig_2 + CPST6$Orig_3) / 3

summary(SEM11, fit.measures = TRUE)
fitMeasures(SEM11, c("chisq", "df", "pvalue"))

#quality
CPST6$Qual_1 <- as.numeric(CPST6$Qual_1)
CPST6$Qual_2 <- as.numeric(CPST6$Qual_2)
CPST6$Qual_3 <- as.numeric(CPST6$Qual_3)


cfa_model12 <-
  '
  LatentQ =~
  Qual_1 +
  Qual_2 +
  Qual_3
 
   LatentQ ~~ 1*LatentQ
  '

SEM12 <- sem(cfa_model12, data = CPST6)
fac.scores12 <- lavPredict(SEM12)
CPST6$FacScoresQ <- c(fac.scores12)

CPST6$MeanQ <- (CPST6$Qual_1 + CPST6$Qual_2 + CPST6$Qual_3) / 3

summary(SEM12, fit.measures = TRUE)
fitMeasures(SEM12, c("chisq", "df", "pvalue"))

CPST6$Dataset <- "CPST6"

CPST6 <- dplyr::select(CPST6, "Problem", "Solutions", "FacScoresQ", "FacScoresO", "MeanQ", "MeanO", "Dataset")



######################
############
#download seventh A DF
#originality
CPST7 <- read_csv("Ralph1.csv")

CPST7 <- data.frame(lapply(CPST7, function(x) ifelse(x == "#NULL!", NA, x)))
CPST7 <- data.frame(lapply(CPST7, function(x) ifelse(x == "#NULL", NA, x)))
CPST7 <- na.omit(CPST7)

CPST7$Orig_1 <- as.numeric(CPST7$Orig_1)
CPST7$Orig_2 <- as.numeric(CPST7$Orig_2)
CPST7$Orig_3 <- as.numeric(CPST7$Orig_3)


cfa_model13 <-
  '
  LatentO =~
  Orig_1 +
  Orig_2 +
  Orig_3
 
   LatentO ~~ 1*LatentO
  '
SEM13 <- sem(cfa_model13, data = CPST7)
fac.scores13 <- lavPredict(SEM13)
CPST7$FacScoresO <- c(fac.scores13)

CPST7$MeanO <- (CPST7$Orig_1 + CPST7$Orig_2 + CPST7$Orig_3) / 3

summary(SEM13, fit.measures = TRUE)
fitMeasures(SEM13, c("chisq", "df", "pvalue"))

#quality
CPST7$Qual_1 <- as.numeric(CPST7$Qual_1)
CPST7$Qual_2 <- as.numeric(CPST7$Qual_2)
CPST7$Qual_3 <- as.numeric(CPST7$Qual_3)


cfa_model14 <-
  '
  LatentQ =~
  Qual_1 +
  Qual_2 +
  Qual_3
 
   LatentQ ~~ 1*LatentQ
  '

SEM14 <- sem(cfa_model14, data = CPST7)
fac.scores14 <- lavPredict(SEM14)
CPST7$FacScoresQ <- c(fac.scores14)

CPST7$MeanQ <- (CPST7$Qual_1 + CPST7$Qual_2 + CPST7$Qual_3) / 3

summary(SEM14, fit.measures = TRUE)
fitMeasures(SEM14, c("chisq", "df", "pvalue"))

CPST7$Dataset <- "CPST7"

CPST7 <- dplyr::select(CPST7, "Problem", "Solutions", "FacScoresQ", "FacScoresO", "MeanQ", "MeanO", "Dataset")



######################
############
#download seventh A DF
#originality
CPST8 <- read_csv("Becky3.csv")

CPST8 <- data.frame(lapply(CPST8, function(x) ifelse(x == "#NULL!", NA, x)))
CPST8 <- na.omit(CPST8)

CPST8$Orig_1 <- as.numeric(CPST8$Orig_1)
CPST8$Orig_2 <- as.numeric(CPST8$Orig_2)
CPST8$Orig_3 <- as.numeric(CPST8$Orig_3)


cfa_model15 <-
  '
  LatentO =~
  Orig_1 +
  Orig_2 +
  Orig_3
 
   LatentO ~~ 1*LatentO
  '
SEM15 <- sem(cfa_model15, data = CPST8)
fac.scores15 <- lavPredict(SEM15)
CPST8$FacScoresO <- c(fac.scores15)

CPST8$MeanO <- (CPST8$Orig_1 + CPST8$Orig_2 + CPST8$Orig_3) / 3

summary(SEM15, fit.measures = TRUE)
fitMeasures(SEM15, c("chisq", "df", "pvalue"))

#quality
CPST8$Qual_1 <- as.numeric(CPST8$Qual_1)
CPST8$Qual_2 <- as.numeric(CPST8$Qual_2)
CPST8$Qual_3 <- as.numeric(CPST8$Qual_3)


cfa_model16 <-
  '
  LatentQ =~
  Qual_1 +
  Qual_2 +
  Qual_3
 
   LatentQ ~~ 1*LatentQ
  '

SEM16 <- sem(cfa_model16, data = CPST8)
fac.scores16 <- lavPredict(SEM16)
CPST8$FacScoresQ <- c(fac.scores16)

CPST8$MeanQ <- (CPST8$Qual_1 + CPST8$Qual_2 + CPST8$Qual_3) / 3

summary(SEM16, fit.measures = TRUE)
fitMeasures(SEM16, c("chisq", "df", "pvalue"))

CPST8$Dataset <- "CPST8"

CPST8 <- dplyr::select(CPST8, "Problem", "Solutions", "FacScoresQ", "FacScoresO", "MeanQ", "MeanO", "Dataset")


######################
############
#download seventh A DF
#originality
CPST9 <- read_csv("Joan1.csv")

CPST9 <- data.frame(lapply(CPST9, function(x) ifelse(x == "#NULL!", NA, x)))
CPST9 <- na.omit(CPST9)

CPST9$Orig_1 <- as.numeric(CPST9$Orig_1)
CPST9$Orig_2 <- as.numeric(CPST9$Orig_2)
CPST9$Orig_3 <- as.numeric(CPST9$Orig_3)


cfa_model17 <-
  '
  LatentO =~
  Orig_1 +
  Orig_2 +
  Orig_3
 
   LatentO ~~ 1*LatentO
  '
SEM17 <- sem(cfa_model17, data = CPST9)
fac.scores17 <- lavPredict(SEM17)
CPST9$FacScoresO <- c(fac.scores17)

CPST9$MeanO <- (CPST9$Orig_1 + CPST9$Orig_2 + CPST9$Orig_3) / 3

summary(SEM17, fit.measures = TRUE)
fitMeasures(SEM17, c("chisq", "df", "pvalue"))


#quality
CPST9$Qual_1 <- as.numeric(CPST9$Qual_1)
CPST9$Qual_2 <- as.numeric(CPST9$Qual_2)
CPST9$Qual_3 <- as.numeric(CPST9$Qual_3)


cfa_model18 <-
  '
  LatentQ =~
  Qual_1 +
  Qual_2 +
  Qual_3
 
   LatentQ ~~ 1*LatentQ
  '

SEM18 <- sem(cfa_model18, data = CPST9)
fac.scores18 <- lavPredict(SEM18)
CPST9$FacScoresQ <- c(fac.scores18)

CPST9$MeanQ <- (CPST9$Qual_1 + CPST9$Qual_2 + CPST9$Qual_3) / 3

summary(SEM18, fit.measures = TRUE)
fitMeasures(SEM18, c("chisq", "df", "pvalue"))

CPST9$Dataset <- "CPST9"

CPST9 <- dplyr::select(CPST9, "Problem", "Solutions", "FacScoresQ", "FacScoresO", "MeanQ", "MeanO", "Dataset")


######################
############
#download seventh A DF
#originality
CPST10 <- read_csv("ACME3.csv")

CPST10 <- data.frame(lapply(CPST10, function(x) ifelse(x == "#NULL!", NA, x)))
CPST10 <- na.omit(CPST10)

CPST10$Orig_1 <- as.numeric(CPST10$Orig_1)
CPST10$Orig_2 <- as.numeric(CPST10$Orig_2)
CPST10$Orig_3 <- as.numeric(CPST10$Orig_3)


cfa_model19 <-
  '
  LatentO =~
  Orig_1 +
  Orig_2 +
  Orig_3
 
   LatentO ~~ 1*LatentO
  '
SEM19 <- sem(cfa_model19, data = CPST10)
fac.scores19 <- lavPredict(SEM19)
CPST10$FacScoresO <- c(fac.scores19)

CPST10$MeanO <- (CPST10$Orig_1 + CPST10$Orig_2 + CPST10$Orig_3) / 3

summary(SEM19, fit.measures = TRUE)
fitMeasures(SEM19, c("chisq", "df", "pvalue"))


#quality
CPST10$Qual_1 <- as.numeric(CPST10$Qual_1)
CPST10$Qual_2 <- as.numeric(CPST10$Qual_2)
CPST10$Qual_3 <- as.numeric(CPST10$Qual_3)


cfa_model20 <-
  '
  LatentQ =~
  Qual_1 +
  Qual_2 +
  Qual_3
 
   LatentQ ~~ 1*LatentQ
  '

SEM20 <- sem(cfa_model20, data = CPST10)
fac.scores20 <- lavPredict(SEM20)
CPST10$FacScoresQ <- c(fac.scores20)

CPST10$MeanQ <- (CPST10$Qual_1 + CPST10$Qual_2 + CPST10$Qual_3) / 3

CPST10$Dataset <- "CPST10"

CPST10 <- dplyr::select(CPST10, "Problem", "Solutions", "FacScoresQ", "FacScoresO", "MeanQ", "MeanO", "Dataset")

summary(SEM20, fit.measures = TRUE)
fitMeasures(SEM20, c("chisq", "df", "pvalue"))


#final DF
CPSTfull <- rbind(CPST1, CPST2, CPST3, CPST4, CPST5, CPST6, CPST7, CPST8, CPST9, CPST10)







