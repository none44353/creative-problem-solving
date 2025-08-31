from nltk.tokenize.punkt import PunktSentenceTokenizer
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
import numpy as np
import string
import time
import torch

def get_dcos_from_features(features):
    # GET DCOS VALUES FOR STORY
    # num = len(features)
    # lower_triangle_indices = np.tril_indices_from(np.random.rand(num, num), k=-1)  
    # dcos_vals = []  
    # for k in range(len(lower_triangle_indices[0])): 
    #     features1 = features[lower_triangle_indices[0][k]]
    #     features2 = features[lower_triangle_indices[1][k]]
    #     dcos = (1 - cos(features1, features2))  
    #     dcos_vals.append(dcos) 
    # story_dcos = torch.mean(torch.stack(dcos_vals)).item()
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
        cos = torch.nn.CosineSimilarity(dim=0)
                
        # CREATE POST-EMBEDDING FILTERING LIST
        filter_list = np.array(['[CLS]', '[PAD]', '[SEP]', '.', ',', '!', '?'])
        
        print("Segmenting sentences and computing DSI...")  # print message to console
        # SEGMENT DATA INTO SENTENCES
        for index, text in enumerate(documents):
            print("Processing ID: " + str(index)) 
            #import pdb; pdb.set_trace()

            # TRAIN SENTENCE SEGEMENTER AND SEGMENT SENTENCE
            segmenter.train(text) # train the segmenter on the text first 
            sentences = segmenter.tokenize(text) # apply the additionally-trained segmenter to the text

            # LOOP OVER SENTENCES AND GET BERT FEATURES (LAYERS 6 & 7)
            features = []  # initialize list to store dcos values, one for each sentence
            words = []
            for i in range(len(sentences)):  # loop over sentences
                sentence = sentences[i].translate(str.maketrans('', '', string.punctuation))
                sent_tokens = tokenizer(sentence, max_length=50, truncation=True, padding='max_length', return_tensors="pt")
                sent_words = [tokenizer.decode([k]) for k in sent_tokens['input_ids'][0]]
                sent_indices = np.where(np.in1d(sent_words, filter_list, invert=True))[0]  # we'll use this to filter out special tokens and punctuation
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
                    # print("Word: " + str(sent_words[j]) + " | Layer 6: " + str(layer6[0,j,:].numpy()) + " | Layer 7: " + str(layer7[0,j,:].numpy()))  # print word and its hidden activation vectors to console

            results.append(get_dcos_from_features(features))
            
    elif model_name == "bge":
        print("Load Model...")
        # INITIALIZE MODEL AND TOKENIZERS
        model = SentenceTransformer('BAAI/bge-large-en-v1.5')  # initialize BGE model
        segmenter = PunktSentenceTokenizer()  # initialize segmenter: does sentence segmentation, returns list
        cos = torch.nn.CosineSimilarity(dim=0)

        print("Segmenting sentences and computing DSI...")  # print message to console
        # SEGMENT DATA INTO SENTENCES
        for index, text in enumerate(documents):
            print("Processing ID: " + str(index)) 

            # TRAIN SENTENCE SEGMENTER AND SEGMENT SENTENCE
            segmenter.train(text)  # train the segmenter on the text first
            sentences = segmenter.tokenize(text)  # apply the additionally-trained segmenter to the text

            # GET BGE SENTENCE FEATURES
            # BGE-large-en is a sentence embedding model, it provides a single vector for a sentence. Therefore, we will get sentence embeddings and calculate DSI based on sentence similarity.
            features = []  # initialize list to store embeddings
            for sentence in sentences:
                if sentence.strip():  # Ensure sentence is not empty or just whitespace
                    # Get the single sentence embedding
                    embedding = model.encode(sentence, convert_to_tensor=True)
                    features.append(embedding)

            results.append(get_dcos_from_features(features))

    elif model_name == "Qwen":
        print("Load Model...")
        # INITIALIZE MODEL AND TOKENIZERS
        model = SentenceTransformer('qwen/Qwen3-embedding-0.6B') # initialize Qwen3-embedding model
        segmenter = PunktSentenceTokenizer() # initialize segmenter: does sentence segmentation, returns list
        cos = torch.nn.CosineSimilarity(dim = 0)

        # CREATE POST-EMBEDDING FILTERING LIST
        filter_list = np.array(['.', ',', '!', '?']) # Qwen doesn't use special tokens like [CLS] and [SEP]

        print("Segmenting sentences and computing DSI...")  # print message to console
        # SEGMENT DATA INTO SENTENCES
        for index, text in enumerate(documents):
            print("Processing ID: " + str(index))  # print current participant ID to console

            # TRAIN SENTENCE SEGEMENTER AND SEGMENT SENTENCE
            segmenter.train(text) # train the segmenter on the text first (unsupervised algorithm that is pretrained and can improve with added training)
            sentences = segmenter.tokenize(text) # apply the additionally-trained segmenter to the text

            # LOOP OVER SENTENCES AND GET QWEN FEATURES
            # Qwen-embedding-0.6B provides a single embedding vector for a sentence, not individual words.
            # Therefore, we will get sentence embeddings. If word-level embeddings are required, a different approach is needed.
            features = []  # initialize list to store dcos values, one for each sentence
            words = []
            
            for sentence in sentences:
                if sentence.strip():  # Ensure sentence is not empty
                    # Get the single sentence embedding
                    embedding = model.encode(sentence, convert_to_tensor=True)
                    # Since we're using sentence embeddings, we'll treat each sentence as a "word" for the DSI calculation
                    words.append(sentence)
                    features.append(embedding)
        
            results.append(get_dcos_from_features(features))

    elapsed_time = time.time() - start_time
    print('elapsed time: ' + str(elapsed_time))
    return results
