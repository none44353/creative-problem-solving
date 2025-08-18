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

print("Loading DSI.py...")  # print message to console

# INITIALIZE MODEL AND TOKENIZERS
#model = SentenceTransformer('qwen/Qwen3-embedding-0.6B') # initialize Qwen3-embedding model
local_model_path = "./Qwen3-embedding-0.6B"
model = SentenceTransformer(local_model_path)

segmenter = PunktSentenceTokenizer() # initialize segmenter: does sentence segmentation, returns list
cos = torch.nn.CosineSimilarity(dim = 0)

print("Loading data...")  # print message to console

# LOAD DATA
d = pd.read_csv(filename, usecols = ["ID","Story"], encoding = "utf-8-sig")  
# d = d.head(1) # you can uncomment this if you want to do a trial run with only the first participant's story instead of the whole dataset
out_df = pd.read_csv(filename, encoding= "utf-8-sig")  # load the original dataframe to merge DSI values back into it later

# CREATE STORAGE DICTIONARY # keys = participant IDs
s = {}

# CREATE POST-EMBEDDING FILTERING LIST
filter_list = np.array(['.', ',', '!', '?']) # Qwen doesn't use special tokens like [CLS] and [SEP]

print("Segmenting sentences and computing QwenDSI...")  # print message to console

# SEGMENT DATA INTO SENTENCES
start_time = time.time()
for index, row in d.iterrows():
    print("Processing ID: " + str(row["ID"]))  # print current participant ID to console
    ID = row["ID"]  # get current participant ID
    text = row["Story"]  # get current story
    s[ID] = {}  # add dict entry for subject and create nested dict to store subject data

    # TRAIN SENTENCE SEGEMENTER AND SEGMENT SENTENCE
    segmenter.train(text) # train the segmenter on the text first (unsupervised algorithm that is pretrained and can improve with added training)
    sentences = segmenter.tokenize(text) # apply the additionally-trained segmenter to the text

    # LOOP OVER SENTENCES AND GET QWEN FEATURES
    features = []  # initialize list to store dcos values, one for each sentence
    words = []
    
    # # Qwen-embedding-0.6B provides a single embedding vector for a sentence, not individual words.
    # # Therefore, we will get sentence embeddings. If word-level embeddings are required, a different approach is needed.
    # for sentence in sentences:
    #     if sentence.strip():  # Ensure sentence is not empty
    #         # Get the single sentence embedding
    #         embedding = model.encode(sentence, convert_to_tensor=True)
            
    #         # Since we're using sentence embeddings, we'll treat each sentence as a "word" for the DSI calculation
    #         words.append(sentence)
    #         features.append(embedding)
    
    for sentence in sentences:
        current_words = sentence.split()
        for word in current_words:
            embedding = model.encode(word, convert_to_tensor=True)
            words.append(word)
            features.append(embedding)

    # GET DCOS VALUES FOR STORY
    num_words = len(words)
    if num_words < 2:  # Cannot compute DSI if there are fewer than 2 sentences
        s[ID]["wordQwenDSI"] = 0.0
        continue
        
    lower_triangle_indices = np.tril_indices_from(np.random.rand(num_words, num_words), k = -1)  # creates a matrix and gets the indices of the lower triangle, omitting diagonal (k = -1)
    
    story_dcos_vals = []  # intialize storage for dcos of current sentence
    for k in range(len(lower_triangle_indices[0])): # loop over lower triangle indices
        features1 = features[lower_triangle_indices[0][k]]
        features2 = features[lower_triangle_indices[1][k]]
        dcos = (1-cos(features1, features2))  # compute dcos
        story_dcos_vals.append(dcos) # store dcos value in list

    mean_story_dcos = torch.mean(torch.stack(story_dcos_vals)).item()  # get average story dcos
    s[ID]["wordQwenDSI"] = mean_story_dcos

print("wordQwenDSI computation complete.")  # print message to console

# MERGE OUTPUT WITH INPUT DATAFRAME
dsi_df = pd.DataFrame.from_dict(s, orient = "index") # make pandas dataframe from DSI dictionary
dsi_df["ID"] = dsi_df.index
out_df = out_df.merge(dsi_df,left_on="ID", right_on="ID") # add wordQwenDSI column to input dataframe
out_df.to_csv('wordQwenDSI_output.csv', index = False) # save updated dataframe
elapsed_time = time.time()-start_time # get elapsed time to compute wordQwenDSI values
print('elapsed time: ' + str(elapsed_time)) # display elapsed time (in seconds)