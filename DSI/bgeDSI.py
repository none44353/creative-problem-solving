# IMPORTS
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import string
import time
import torch

# USER EDIT
filename = "qa_data.csv"

print("Loading bgeDSI.py...")  # print message to console

# INITIALIZE MODEL AND TOKENIZERS
model = SentenceTransformer('BAAI/bge-large-en-v1.5')  # initialize BGE model
segmenter = PunktSentenceTokenizer()  # initialize segmenter: does sentence segmentation, returns list
cos = torch.nn.CosineSimilarity(dim=0)

print("Loading data...")  # print message to console

# LOAD DATA
d = pd.read_csv(filename, usecols=["ID", "Story"], encoding="utf-8-sig")
# d = d.head(1) # you can uncomment this if you want to do a trial run with only the first participant's story instead of the whole dataset
out_df = pd.read_csv(filename, encoding="utf-8-sig")  # load the original dataframe to merge DSI values back into it later

# CREATE STORAGE DICTIONARY # keys = participant IDs
s = {}

# CREATE POST-EMBEDDING FILTERING LIST
# BGE is a sentence-level model, so this filter list is not used for word tokens.
# We will filter out empty sentences instead.

print("Segmenting sentences and computing DSI...")  # print message to console

# SEGMENT DATA INTO SENTENCES
start_time = time.time()
for index, row in d.iterrows():
    print("Processing ID: " + str(row["ID"]))  # print current participant ID to console
    ID = row["ID"]  # get current participant ID
    text = row["Story"]  # get current story
    s[ID] = {}  # add dict entry for subject and create nested dict to store subject data

    # TRAIN SENTENCE SEGMENTER AND SEGMENT SENTENCE
    segmenter.train(text)  # train the segmenter on the text first
    sentences = segmenter.tokenize(text)  # apply the additionally-trained segmenter to the text

    # GET BGE SENTENCE FEATURES
    features = []  # initialize list to store embeddings
    
    # BGE-large-en is a sentence embedding model, it provides a single vector for a sentence.
    # Therefore, we will get sentence embeddings and calculate DSI based on sentence similarity.    
    # for sentence in sentences:
    #     if sentence.strip():  # Ensure sentence is not empty or just whitespace
    #         # Get the single sentence embedding
    #         embedding = model.encode(sentence, convert_to_tensor=True)
    #         features.append(embedding)

    for sentence in sentences:
        words = sentence.split()
        for word in words:
            embedding = model.encode(word, convert_to_tensor=True)
            features.append(embedding)

    # GET DCOS VALUES FOR STORY
    num_sentences = len(features)
    if num_sentences < 2:  # Cannot compute DSI if there are fewer than 2 sentences
        s[ID]["wordbgeDSI"] = 0.0
        continue

    # Create a matrix for the lower triangle indices to calculate pairwise similarity
    lower_triangle_indices = np.tril_indices_from(np.random.rand(num_sentences, num_sentences), k=-1)
    
    story_dcos_vals = []  # intialize storage for dcos of current story
    for k in range(len(lower_triangle_indices[0])):  # loop over lower triangle indices
        features1 = features[lower_triangle_indices[0][k]]
        features2 = features[lower_triangle_indices[1][k]]
        dcos = (1 - cos(features1, features2))  # compute dcos
        story_dcos_vals.append(dcos)  # store dcos value in list

    mean_story_dcos = torch.mean(torch.stack(story_dcos_vals)).item()  # get average story dcos
    s[ID]["wordbgeDSI"] = mean_story_dcos

print("wordbgeDSI computation complete.")  # print message to console

# MERGE OUTPUT WITH INPUT DATAFRAME
dsi_df = pd.DataFrame.from_dict(s, orient="index")  # make pandas dataframe from DSI dictionary
dsi_df["ID"] = dsi_df.index
out_df = out_df.merge(dsi_df, left_on="ID", right_on="ID")  # add wordbgeDSI column to input dataframe
out_df.to_csv('wordbgeDSI_output.csv', index=False)  # save updated dataframe
elapsed_time = time.time() - start_time  # get elapsed time to compute wordbgeDSI values
print('elapsed time: ' + str(elapsed_time))  # display elapsed time (in seconds)