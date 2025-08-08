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

print("Loading DSI.py...") # print message to console

# INITIALIZE BERT AND TOKENIZERS
model = BertModel.from_pretrained("bert-large-uncased", output_hidden_states = True) # initialize BERT model instance
model.eval()
segmenter = PunktSentenceTokenizer() # initialize segmenter: does sentence segmentation, returns list
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased') # initialize BERT tokenizer
cos = torch.nn.CosineSimilarity(dim = 0)

print("Loading data...") # print message to console

# LOAD DATA
d = pd.read_csv(filename, usecols = ["ID","Story", "ProblemID"], encoding = "utf-8-sig")
# d = d.head(1) # you can uncomment this if you want to do a trial run with only the first participant's story instead of the whole dataset
out_df = pd.read_csv(filename, encoding= "utf-8-sig") # load the original dataframe to merge DSI values back into it later

# CREATE STORAGE DICTIONARY # keys = participant IDs
s = {}

# CREATE POST-EMBEDDING FILTERING LIST
filter_list = np.array(['[CLS]', '[PAD]', '[SEP]', '.', ',', '!', '?'])

print("Segmenting sentences and computing peerDSI...") # print message to console

problem_ids = d["ProblemID"].unique()
print("Unique ProblemIDs:", problem_ids)

def get_words_and_features(sentences):
    # LOOP OVER SENTENCES AND GET BERT FEATURES (LAYERS 6 & 7)
    features = [] # initialize list to store dcos values, one for each sentence
    words = []
    for i in range(len(sentences)): # loop over sentences
        sentence = sentences[i].translate(str.maketrans('','',string.punctuation))
        sent_tokens = tokenizer(sentence, max_length = 50, truncation = True, padding = 'max_length', return_tensors="pt")
        sent_words = [tokenizer.decode([k]) for k in sent_tokens['input_ids'][0]]
        sent_indices = np.where(np.in1d(sent_words, filter_list, invert = True))[0] # we'll use this to filter out special tokens and punctuation
        with torch.no_grad():
            sent_output = model(**sent_tokens) # feed model the sentence tokens and get outputs
            hids = sent_output.hidden_states # isolate hidden layer activations
        layer6 = hids[6] # isolate layer 6 hidden activations
        layer7 = hids[7] # do the same for layer 7

        for j in sent_indices: # loop over words and create list of all hidden vectors from layers 6 & 7; additionally store number of words (doubled, to account for layer 6 and 7 duplicates)
            words.append(sent_words[j])
            words.append(sent_words[j])
            features.append(layer6[0,j,:]) # layer 6 features
            features.append(layer7[0,j,:]) # layer 7 features
            
    return words, features

intermediate_results = {}

# SEGMENT DATA INTO SENTENCES
start_time = time.time()
for problem_id in problem_ids:
    problem_data = d[d["ProblemID"] == problem_id]
    print(f"Processing ProblemID: {problem_id} with {len(problem_data)} entries")
    with open(f"../Data/problem/{problem_id}.txt", "r", encoding="utf-8") as f:
        prompt = f.read()
    
    segmenter.train(prompt)
    sentences = segmenter.tokenize(prompt)
    prompt_words, prompt_features = get_words_and_features(sentences)
    
    total_features = []
    for index, row in problem_data.iterrows():
        print("Processing ID: " + str(row["ID"]))
        ID = row["ID"]
        text = row["Story"]
        s[ID] = {}
        segmenter.train(text)
        sentences = segmenter.tokenize(text)
        words, features = get_words_and_features(sentences)
        intermediate_results[ID] = {
            "words": words,
            "features": features
            }
        total_features.extend(features)
        print(f"Total words processed for ProblemID {problem_id}: {len(total_features)}")
        
    for index, row in problem_data.iterrows():
        ID = row["ID"]
        words = intermediate_results[ID]["words"]
        features = intermediate_results[ID]["features"]
        print("Processing ID: " + str(row["ID"]) + "Current Number of Features: " + str(len(features)))

        peer_dos_vals = []
        for feature1 in features:
            for feature2 in total_features:
                dcos = (1 - cos(feature1, feature2))
                peer_dos_vals.append(dcos)
        mean_peer_dos = torch.mean(torch.stack(peer_dos_vals)).item()
        s[ID]["peer-DSI"] = mean_peer_dos
        
        prompt_dos_vals = []
        for feature1 in features:
            for feature2 in prompt_features:
                dcos = (1 - cos(feature1, feature2))
                prompt_dos_vals.append(dcos)
        mean_prompt_dos = torch.mean(torch.stack(prompt_dos_vals)).item()
        s[ID]["prompt-DSI"] = mean_prompt_dos

print("DSI computation complete.") # print message to console

# MERGE OUTPUT WITH INPUT DATAFRAME
dsi_df = pd.DataFrame.from_dict(s, orient = "index") # make pandas dataframe from DSI dictionary
dsi_df["ID"] = dsi_df.index
out_df = out_df.merge(dsi_df,left_on="ID", right_on="ID") # add DSI column to input dataframe
out_df.to_csv('final.csv', index = False) # save updated dataframe
elapsed_time = time.time()-start_time # get elapsed time to compute DSI values
print('elapsed time: ' + str(elapsed_time)) # display elapsed time (in seconds)