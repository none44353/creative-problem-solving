import torch
import string
import numpy as np

from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from nltk.tokenize.punkt import PunktSentenceTokenizer


def compute_bert_dsi(text_list, verbose=False):
    # Load BERT pre-trained model, and set it to test mode
    model = BertModel.from_pretrained("bert-large-uncased", output_hidden_states = True) # initialize BERT model instance
    model.eval()
    
    # Initialize segmenter, tokenizer, and cosine similarity calculator
    segmenter = PunktSentenceTokenizer() # initialize segmenter: does sentence segmentation, returns list
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased') # initialize BERT tokenizer
    cos = torch.nn.CosineSimilarity(dim=0)
    
    # Create Post-embedding filtering list, and the list to store DSI.
    filter_list = np.array(['[CLS]', '[PAD]', '[SEP]', '.', ',', '!', '?'])
    dsi_list = []
    
    for text in tqdm(text_list):
         # Train sentence segmenter, and segment sentence
        segmenter.train(text)
        sentences = segmenter.tokenize(text)
        
        # Loop over sentences and get BERT features (Layers 6 & 7)
        words = []
        features = []  # initialize list to store dcos values, one for each sentence
        
        for i in range(len(sentences)):
            sentence = sentences[i].translate(str.maketrans('', '', string.punctuation))
            sent_tokens = tokenizer(sentence, max_length=50, truncation=True, padding='max_length', return_tensors="pt")
            sent_words = [tokenizer.decode([k]) for k in sent_tokens['input_ids'][0]]
            sent_indices = np.where(np.isin(sent_words, filter_list, invert=True))[0]  # This will be used to filter out special tokens and punctuation
            with torch.no_grad():
                sent_output = model(**sent_tokens) # feed model the sentence tokens and get outputs
                hids = sent_output.hidden_states # isolate hidden layer activations
            layer6 = hids[6] # isolate layer 6 hidden activations
            layer7 = hids[7] # do the same for layer 7

            for j in sent_indices:  
                # loop over words and create list of all hidden vectors from layers 6 & 7; 
                # additionally store number of words (doubled, to account for layer 6 and 7 duplicates)
                words.append(sent_words[j])
                words.append(sent_words[j])
                features.append(layer6[0, j, :])  # layer 6 features
                features.append(layer7[0, j, :])  # layer 7 features
                if verbose:
                    print("Word: " + str(sent_words[j]) + " | Layer 6: " + str(layer6[0, j, :].numpy()) + " | Layer 7: " + str(layer7[0, j, :].numpy()))  # print word and its hidden activation vectors to console

        # Get docs values for the text
        num_words = len(words) # number of words, in terms of hidden activation vectors (2*words)
        lower_triangle_indices = np.tril_indices_from(np.random.rand(num_words, num_words), k=-1)  # creates a matrix that represents words*2 (i.e., from word representations from both layer 6+7) and gets the indices of the lower triangle, omitting diagonal (k = -1)A
        
        story_dcos_vals = []  # intialize storage for dcos of current sentence
        for k in range(len(lower_triangle_indices[0])): # loop over lower triangle indices
            features1 = features[lower_triangle_indices[0][k]]
            features2 = features[lower_triangle_indices[1][k]]
            dcos = (1 - cos(features1, features2))  # compute dcos
            story_dcos_vals.append(dcos) # store dcos value in list

        mean_story_dcos = torch.mean(torch.stack(story_dcos_vals)).item()  # get average story dcos
        dsi_list.append(mean_story_dcos)  # append the DSI value to the list
    
    return dsi_list
