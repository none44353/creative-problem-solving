#Join all DFs

PredictedTestSet <- dplyr::select(PredictedTestSet, -1)
PredictedHeldoutSet <- dplyr::select(PredictedHeldoutSet, -1)

PredictedTestSet$set <- "test"
PredictedHeldoutSet$set <- "heldout"

oGPT <- rbind(PredictedTestSet, PredictedHeldoutSet)

names(oGPT)[2] <- "labelOriginality"
names(oGPT)[3] <- "predictionOriginality_GPT2"



PredictedTestSet <- dplyr::select(PredictedTestSet, -1)
PredictedHeldoutSet <- dplyr::select(PredictedHeldoutSet, -1)

PredictedTestSet$set <- "test"
PredictedHeldoutSet$set <- "heldout"

qGPT <- rbind(PredictedTestSet, PredictedHeldoutSet)

names(qGPT)[2] <- "labelQuality"
names(qGPT)[3] <- "predictionQuality_GPT2"

GPT <- left_join(qGPT, oGPT)

########
########


PredictedTestSet <- dplyr::select(PredictedTestSet, -1)
PredictedHeldoutSet <- dplyr::select(PredictedHeldoutSet, -1)

PredictedTestSet$set <- "test"
PredictedHeldoutSet$set <- "heldout"

oRoBERTa <- rbind(PredictedTestSet, PredictedHeldoutSet)

names(oRoBERTa)[2] <- "labelOriginality"
names(oRoBERTa)[3] <- "predictionOriginality_RoBERTa"



PredictedTestSet <- dplyr::select(PredictedTestSet, -1)
PredictedHeldoutSet <- dplyr::select(PredictedHeldoutSet, -1)

PredictedTestSet$set <- "test"
PredictedHeldoutSet$set <- "heldout"

qRoBERTa <- rbind(PredictedTestSet, PredictedHeldoutSet)

names(qRoBERTa)[2] <- "labelQuality"
names(qRoBERTa)[3] <- "predictionQuality_RoBERTa"

RoBERTa <- left_join(qRoBERTa, oRoBERTa)

######
######

AllModelsCPST <- left_join(GPT, RoBERTa)


AllModelsCPST$prederrorQualGPT2 <- AllModelsCPST$labelQuality - AllModelsCPST$predictionQuality_GPT2
AllModelsCPST$prederrorQualRoBERTa <- AllModelsCPST$labelQuality - AllModelsCPST$predictionQuality_RoBERTa
AllModelsCPST$prederrorOrigGPT2 <- AllModelsCPST$labelOriginality - AllModelsCPST$predictionOriginality_GPT2
AllModelsCPST$prederrorOrigRoBERTa <- AllModelsCPST$labelOriginality - AllModelsCPST$predictionOriginality_RoBERTa

RLPS <- subset(CPSTfulldataset2, set == "test" | set == "heldout")

RLPS2 <- dplyr::select(RLPS, "Solutions", "ProblemID", "set", "wordcount", "DSI")
names(RLPS2)[1] <- "text"

AllModelsCPST <- left_join(RLPS2, AllModelsCPST)

#AllModelsCPST$OpenScoringFormat <- paste(AllModelsCPST$ProblemID, AllModelsCPST$text, sep = ", ")

write_csv(AllModelsCPST, file = "AllModelsCPST.csv")

