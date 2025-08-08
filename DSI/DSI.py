"""

All code in this file is licensed to John D. Patterson from The Pennsylvania State University, 04-04-2022, under the Creative Commons Attribution-NonCommerical-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

Link to License Deed https://creativecommons.org/licenses/by-nc-sa/4.0/

Link to Legal Code https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

Please cite Johnson, D. R., Kaufman, J. C., Baker, B. S., Barbot, B., Green, A., van Hell, J., … Beaty, R. (2021, December 1). Extracting Creativity from Narratives using Distributional Semantic Modeling. Retrieved from psyarxiv.com/fmwgy in any publication or presentation

"""

# IMPORTS
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from transformers import BertTokenizer, BertModel, BertConfig
import numpy as np
import pandas as pd
import string
import time
import torch

# USER EDIT
filename = "qa_data.csv"

print("Loading DSI.py...")  # print message to console

# INITIALIZE BERT AND TOKENIZERS
model = BertModel.from_pretrained("bert-large-uncased", output_hidden_states = True) # initialize BERT model instance
model.eval()
segmenter = PunktSentenceTokenizer() # initialize segmenter: does sentence segmentation, returns list
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased') # initialize BERT tokenizer
cos = torch.nn.CosineSimilarity(dim = 0)

print("Loading data...")  # print message to console

# LOAD DATA
d = pd.read_csv(filename, usecols = ["ID","Story"], encoding = "utf-8-sig")  
# d = d.head(1) # you can uncomment this if you want to do a trial run with only the first participant's story instead of the whole dataset
out_df = pd.read_csv(filename, encoding= "utf-8-sig")  # load the original dataframe to merge DSI values back into it later

# CREATE STORAGE DICTIONARY # keys = participant IDs
s = {}

# CREATE POST-EMBEDDING FILTERING LIST
filter_list = np.array(['[CLS]', '[PAD]', '[SEP]', '.', ',', '!', '?'])

print("Segmenting sentences and computing DSI...")  # print message to console

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

    # LOOP OVER SENTENCES AND GET BERT FEATURES (LAYERS 6 & 7)
    features = []  # initialize list to store dcos values, one for each sentence
    words = []
    for i in range(len(sentences)):  # loop over sentences
        sentence = sentences[i].translate(str.maketrans('','',string.punctuation))
        sent_tokens = tokenizer(sentence, max_length = 50, truncation = True, padding = 'max_length', return_tensors="pt")
        sent_words = [tokenizer.decode([k]) for k in sent_tokens['input_ids'][0]]
        sent_indices = np.where(np.in1d(sent_words, filter_list, invert = True))[0]  # we'll use this to filter out special tokens and punctuation
        with torch.no_grad():
            sent_output = model(**sent_tokens) # feed model the sentence tokens and get outputs
            hids = sent_output.hidden_states # isolate hidden layer activations
        layer6 = hids[6] # isolate layer 6 hidden activations
        layer7 = hids[7] # do the same for layer 7

        for j in sent_indices:  # loop over words and create list of all hidden vectors from layers 6 & 7; additionally store number of words (doubled, to account for layer 6 and 7 duplicates)
            words.append(sent_words[j])
            words.append(sent_words[j])
            features.append(layer6[0,j,:])  # layer 6 features
            features.append(layer7[0,j,:])  # layer 7 features
            print("Word: " + str(sent_words[j]) + " | Layer 6: " + str(layer6[0,j,:].numpy()) + " | Layer 7: " + str(layer7[0,j,:].numpy()))  # print word and its hidden activation vectors to console

    # GET DCOS VALUES FOR STORY
    num_words = len(words) # number of words, in terms of hidden activation vectors (2*words)
    lower_triangle_indices = np.tril_indices_from(np.random.rand(num_words, num_words), k = -1)  # creates a matrix that represents words*2 (i.e., from word representations from both layer 6+7) and gets the indices of the lower triangle, omitting diagonal (k = -1)A
    
    story_dcos_vals = []  # intialize storage for dcos of current sentence
    for k in range(len(lower_triangle_indices[0])): # loop over lower triangle indices
        features1 = features[lower_triangle_indices[0][k]]
        features2 = features[lower_triangle_indices[1][k]]
        dcos = (1-cos(features1, features2))  # compute dcos
        story_dcos_vals.append(dcos) # store dcos value in list

    mean_story_dcos = torch.mean(torch.stack(story_dcos_vals)).item()  # get average story dcos
    s[ID]["DSI"] = mean_story_dcos


print("DSI computation complete.")  # print message to console

# MERGE OUTPUT WITH INPUT DATAFRAME
dsi_df = pd.DataFrame.from_dict(s, orient = "index") # make pandas dataframe from DSI dictionary
dsi_df["ID"] = dsi_df.index
out_df = out_df.merge(dsi_df,left_on="ID", right_on="ID") # add DSI column to input dataframe
out_df.to_csv('DSI_output.csv', index = False) # save updated dataframe
elapsed_time = time.time()-start_time # get elapsed time to compute DSI values
print('elapsed time: ' + str(elapsed_time)) # display elapsed time (in seconds)
