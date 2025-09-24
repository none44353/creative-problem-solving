import time
import torch
import string
import numpy as np

from tqdm import tqdm
from nltk.tokenize.punkt import PunktSentenceTokenizer
from transformers import AutoModel, AutoTokenizer, BertTokenizer, BertModel


def get_dcos_from_features(features):
    # GET DCOS VALUES FOR STORY
    if len(features) < 2:
        return 0.0
    
    normalized_features = torch.stack([f / f.norm() for f in features])
    cosine_sim_matrix = normalized_features @ normalized_features.T
    lower_triangle_values = cosine_sim_matrix.triu(diagonal=1)
    average_similarity = lower_triangle_values[lower_triangle_values != 0].mean()
    mean_dcos = float(1 - average_similarity)
    return mean_dcos


def get_dsi(model_name, documents):
    print("Starting DSI computation...")
    start_time = time.time()
    results = []
    
    if model_name == "bert":
        print("Load Model...")
        # INITIALIZE BERT AND TOKENIZERS
        model = BertModel.from_pretrained("bert-large-uncased", output_hidden_states = True) 
        model.eval()
        segmenter = PunktSentenceTokenizer()
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
                
        # CREATE POST-EMBEDDING FILTERING LIST
        filter_list = np.array(['[CLS]', '[PAD]', '[SEP]', '.', ',', '!', '?'])
        
        print("Segmenting sentences and computing DSI...")  # print message to console
        # SEGMENT DATA INTO SENTENCES
        for index, text in enumerate(tqdm(documents)):
            # TRAIN SENTENCE SEGEMENTER AND SEGMENT SENTENCE
            segmenter.train(text) # train the segmenter on the text first 
            sentences = segmenter.tokenize(text) # apply the additionally-trained segmenter to the text
            sentences = [subsentence for sentence in sentences for subsentence in sentence.split('\n')] # further split sentences by newline characters to avoid possible calculation problems when adding extra lines

            # LOOP OVER SENTENCES AND GET BERT FEATURES (LAYERS 6 & 7)
            features = []  # initialize list to store dcos values, one for each sentence
            for i in range(len(sentences)):  # loop over sentences
                sentence = sentences[i].translate(str.maketrans('', '', string.punctuation))
                sent_tokens = tokenizer(sentence, max_length=50, truncation=True, padding='max_length', return_tensors="pt")
                sent_words = [tokenizer.decode([k]) for k in sent_tokens['input_ids'][0]]
                sent_indices = np.where(np.isin(sent_words, filter_list, invert=True))[0]  # we'll use this to filter out special tokens and punctuation
                with torch.no_grad():
                    sent_output = model(**sent_tokens) # feed model the sentence tokens and get outputs
                    hids = sent_output.hidden_states # isolate hidden layer activations
                layer6 = hids[6] # isolate layer 6 hidden activations
                layer7 = hids[7] # do the same for layer 7

                for j in sent_indices:  # loop over words and create list of all hidden vectors from layers 6 & 7; additionally store number of words (doubled, to account for layer 6 and 7 duplicates)
                    features.append(layer6[0,j,:])  # layer 6 features
                    features.append(layer7[0,j,:])  # layer 7 features
            
            # GET DCOS VALUES FOR STORY
            results.append(get_dcos_from_features(features))
            
    elif model_name == "bge" or model_name == "Qwen":
        # INITIALIZE MODEL AND TOKENIZERS
        print("Load Model...")
        official_model_name = "BAAI/bge-large-en-v1.5" if model_name == "bge" else "qwen/Qwen3-embedding-0.6B"
        model = AutoModel.from_pretrained(official_model_name, output_hidden_states=True) # initialize model instance
        model.eval() # Turn the model to evaluation mode
        segmenter = PunktSentenceTokenizer() # initialize segmenter: does sentence segmentation, returns list
        tokenizer = AutoTokenizer.from_pretrained(official_model_name) # initialize tokenizer
        
        # CREATE POST-EMBEDDING FILTERING LIST
        filter_list = np.array(tokenizer.all_special_tokens + ['.', ',', '!', '?', ':'])

        print("Segmenting sentences and computing DSI...")  # print message to console
        
        # SEGMENT DATA INTO SENTENCES
        start_time = time.time()
        for text in tqdm(documents):
            # TRAIN SENTENCE SEGEMENTER AND SEGMENT SENTENCE
            segmenter.train(text) # train the segmenter on the text first (unsupervised algorithm that is pretrained and can improve with added training)
            sentences = segmenter.tokenize(text) # apply the additionally-trained segmenter to the text
            sentences = [subsentence for sentence in sentences for subsentence in sentence.split('\n')] # further split sentences by newline characters to avoid possible calculation problems when adding extra lines

            # LOOP OVER SENTENCES AND GET FEATURES (For BGE: Layers 6 & 7; For Qwen: Layers 7 & 8)
            features = []  # initialize list to store dcos values, one for each sentence
            for i in range(len(sentences)):  # loop over sentences
                sentence = sentences[i].translate(str.maketrans('', '', string.punctuation))
                sent_tokens = tokenizer(sentence, max_length=50, truncation=True, padding='max_length', return_tensors="pt")
                sent_words = [tokenizer.decode([k]) for k in sent_tokens['input_ids'][0]]
                sent_indices = np.where(np.isin(sent_words, filter_list, invert=True))[0]  # we'll use this to filter out special tokens and punctuation
                with torch.no_grad():
                    sent_output = model(**sent_tokens) # feed model the sentence tokens and get outputs
                    hids = sent_output.hidden_states # isolate hidden layer activations
                if model_name == "bge":
                    layer6 = hids[6] # isolate layer 6 hidden activations
                    layer7 = hids[7] # do the same for layer 7
                elif model_name == "Qwen":
                    layer7 = hids[7] # isolate layer 7 hidden activations
                    layer8 = hids[8] # do the same for layer 8

                for j in sent_indices:  # loop over words and create list of all hidden vectors from specified layers.
                    if model_name == "bge":
                        features.append(layer6[0,j,:])  # layer 6 features
                        features.append(layer7[0,j,:])  # layer 7 features
                    elif model_name == "Qwen":
                        features.append(layer7[0,j,:])  # layer 7 features
                        features.append(layer8[0,j,:])  # layer 8 features

            # GET DCOS VALUES FOR STORY
            results.append(get_dcos_from_features(features))
    
    elapsed_time = time.time() - start_time
    print('elapsed time: ' + str(elapsed_time))
    return results
