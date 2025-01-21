#Avantika Shrestha 2024, based on code from Ermal Toto and Ricardo Flores

from AudiBERTutils import recordRun, pad_sequences, str2bool, runCount
import random
import time
import datetime
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef, f1_score, auc, recall_score, precision_score, accuracy_score, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

#Writing out some useful functions
def flat_accuracy(preds, labels):
    '''
    Function to calculate the accuracy of our predictions vs labels
    '''
    
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def upsample(df, criteria):
    pos_class = df[df[criteria]==1]
    neg_class = df[df[criteria]==0]
    
    if len(pos_class) > len(neg_class):
        majority = pos_class
        minority = neg_class
    elif len(pos_class) < len(neg_class):
        majority = neg_class
        minority = pos_class
    else:
        return df
    
    upsample = resample(minority,
                        replace=True,
                        n_samples=len(majority),
                        random_state=seed_val)
    
    df = pd.concat([majority, upsample])
    return df

#Reading in data
data = pd.read_csv('', index_col=0)

audio1 = data[[]]
audio2 = data[[]]

#Default Configuration
MAX_LEN = 128
lr = 2e-5
batch_size = 1
useAudio = False
useGender = False
epochs = 10
question = ''
typeOfTask = ''
follow_up_question = False
logtodb = False
maxruns = 10
modelName = ''
save_model = False

if 'audio1' in question:
    wav_location = ''
elif 'audio2' in question:
    wav_location = ''

if modelName == 'VGG':
    numVectors = 3
else:
    numVectors = 15

useLSTM = True
useLastVectors = False
VGGSeconds = numVectors
numPaseFilters = numVectors

detectGender = False
if detectGender:
    useGender = False
    
num_gender_features = 1
if not useGender:
    num_gender_features = 0

#Setting up arguments for running experiments
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, help="Number of Epochs to train. Default=4")
parser.add_argument("--question", help="DAIC Question Abreviation (Folder Name). Default=feeling_lately")
parser.add_argument("--modelName", help="Model Name. Default=RNN")
parser.add_argument("--logtodb", type=str2bool, help="Log Run To DB. Default=True")
parser.add_argument("--useAudio", type=str2bool, help="Use useAudio. Default=True")
parser.add_argument("--maxruns", type=int, help="Maximum number of runs for same configuraiton Default=20")

args = parser.parse_args()

if args.__dict__["epochs"]  is not None:
    epochs = args.__dict__["epochs"]
if args.__dict__["useAudio"]  is not None:
    useAudio = args.__dict__["useAudio"]
if args.__dict__["question"]  is not None:
    question = args.__dict__["question"]
if args.__dict__["modelName"]  is not None:
    modelName = args.__dict__["modelName"] 
if args.__dict__["logtodb"]  is not None:
    logtodb = args.__dict__["logtodb"]     
if args.__dict__["maxruns"]  is not None:
    maxruns = args.__dict__["maxruns"] 
    
#Assigning databse for collection of results
db = ''
    
print('typeOfTask:', typeOfTask)

#Logging experiment details
configuration = {
    'MAX_LEN': MAX_LEN,
    'batch_size': batch_size,
    'model': modelName,
    'typeOfTask': typeOfTask,
    'useAudio':useAudio,
    'useGender':useGender,
    'epochs':epochs,
    'question':question,
    'numPaseFilters': numPaseFilters,
    'numVectors': numVectors,
    'useLastVectors': useLastVectors,
    'lr':lr
}

short_configuration = ''
for configValue in configuration:
    short_configuration = short_configuration + str(configuration[configValue])

print('short_configuration:', short_configuration)
results = {}

results['run_id'] = runCount(short_configuration, db=db)

if results['run_id'] > maxruns:
    print("Exiting. There are too many experiments with this configuraiton. Increase Max Run Count")
    sys.exit(0)
print('results[\'run_id\']:', results['run_id'])

#Setting device to GPU or CPU, depending on resources
if torch.cuda.is_available():    
    #Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

#Loading model
from transformers import AdamW

if modelName.upper() == 'BERT':
    from models.BERT import BERTBase
    model = BERTBase.from_pretrained("bert-base-uncased", 
                                     num_labels = 2, 
                                     output_attentions = False, 
                                     output_hidden_states = False, 
                                     num_gender_features = num_gender_features,
                                     numPaseFilters = numPaseFilters)  

if modelName.upper() == 'BERT_VGG':
    from torchvggish2.vggish import VGGish
    from models.BERTVGGishDualAttention import BERTVGGishDualAttention
    model = BERTVGGishDualAttention.from_pretrained("bert-base-uncased",
                                                     num_labels = 2,
                                                     output_attentions = False,
                                                     output_hidden_states = False,
                                                     num_gender_features = num_gender_features)

if modelName.upper() == 'VGG':
    from torchvggish2.vggish import VGGish
    from models.VGGish import VGGForClassificationWithAudio    
    model = VGGForClassificationWithAudio(num_gender_features = num_gender_features, 
                                          numVectors = numVectors, 
                                          useLSTM = useLSTM, 
                                          useLastVectors = useLastVectors)

#Tell pytorch to run this model on the device
model.to(device)

#Setting seed value for experiments
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

#Train-Test Split
if 'audio1' in question:
    train_data, test_data = train_test_split(audio1, test_size=0.2, random_state=seed_val)
elif 'audio2' in question:
    train_data, test_data = train_test_split(audio2, test_size=0.2, random_state=seed_val)

print('train_data:', len(train_data))
print('test_data:', len(test_data))

#Loading BERT Tokenizer
if 'BERT' in modelName.upper():
    from transformers import BertTokenizer
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    
#Train set preparation
if 'upsample' in typeOfTask:
    print('UPSAMPLED')
    
    if 'audio1' in question:
        if 'phq' in question:
            train_data = train_data[['audio1_x', 'phq']]
            train_data = upsample(train_data, 'phq')

            sentences = train_data['audio1_x'].values 
            labels = train_data['phq'].values
        
        elif 'gad' in question:
            train_data = train_data[['audio1_x', 'gad']]
            train_data = upsample(train_data, 'ygad')

            sentences = train_data['audio1_x'].values 
            labels = train_data['gad'].values    
        
        else:
            train_data = train_data[['audio1_x', 'q9']]
            train_data = upsample(train_data, 'q9')

            sentences = train_data['audio1_x'].values 
            labels = train_data['q9'].values

    if 'audio2' in question:
        if 'phq' in question:
            train_data = train_data[['audio2_x', 'phq']]
            train_data = upsample(train_data, 'phq')

            sentences = train_data['audio2_x'].values 
            labels = train_data['phq'].values
        
        elif 'gad' in question:
            train_data = train_data[['audio2_x', 'gad']]
            train_data = upsample(train_data, 'gad')

            sentences = train_data['audio2_x'].values 
            labels = train_data['gad'].values    
        
        else:
            train_data = train_data[['audio2_x', 'q9']]
            train_data = upsample(train_data, 'q9')

            sentences = train_data['audio2_x'].values 
            labels = train_data['q9'].values
    
    train_identifiers = train_data.index.array

else:
    if 'audio1' in question:
        sentences = train_data['audio1_x'].values 
    elif 'audio2' in question:
        sentences = train_data['audio2_x'].values 

    if 'phq' in question:
        labels = train_data['phq'].values
    elif 'gad' in question:
        labels = train_data['gad'].values
    else:
        labels = train_data['q9'].values

    train_identifiers = train_data.index.array

if 'BERT' in modelName.upper():
#Tokenizing the train set
    input_ids = []
    line = 0
    
    for sentence in sentences:
        encoded_sent = tokenizer.encode(sentence, add_special_tokens = True)
        input_ids.append(encoded_sent)
        line = line + 1
        
    print('Original: ', sentences[0])
    print('Token IDs:', input_ids[0])
    
if 'BERT' in modelName.upper():
#Padding the train set
    print('\nPadding/truncating all sentences to %d values...' % MAX_LEN)
    print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))

    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")

    print('\nDone.')
    
if 'BERT' in modelName.upper():
#Attention mask for the train set
    attention_masks = []
    
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)

#Turning train dataset into a tensor dataset        
if 'BERT' in modelName.upper():
    train_inputs = torch.tensor(input_ids)
    train_masks = torch.tensor(attention_masks)

train_labels = torch.tensor(labels) 
train_identifiers = torch.tensor([int(id) for id in train_identifiers])

if 'BERT' in modelName.upper():
    train = TensorDataset(train_inputs, train_masks, train_labels, train_identifiers)
else:
    train = TensorDataset(train_labels, train_identifiers)

train_sampler = RandomSampler(train)
train_dataloader = DataLoader(train, sampler=train_sampler, batch_size=batch_size) #Train dataloader with random sampler and batched by batch size

#Test set preperation
test_identifiers = test_data.index.array

if 'audio1' in question:
    sentences = test_data['audio1_x']
elif 'audio2' in question:
    sentences = test_data['audio2_x']

if 'phq' in question:
    labels = test_data['phq'].values
elif 'gad' in question:
    labels = test_data['gad'].values
else:
    labels = test_data['q9'].values

if 'BERT' in modelName.upper():
#Tokenizing the test set
    input_ids = []
    line = 0
    
    for sentence in sentences:
        encoded_sent = tokenizer.encode(sentence, add_special_tokens = True)
        input_ids.append(encoded_sent)
        line = line + 1
    
if 'BERT' in modelName.upper():
#Padding the test set
    print('\nPadding/truncating all sentences to %d values...' % MAX_LEN)
    print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))

    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")

    print('\nDone.')
    
if 'BERT' in modelName.upper():
#Attention mask for the test set
    attention_masks = []
    
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)

#Turning train dataset into a tensor dataset           
if 'BERT' in modelName.upper():
    test_inputs = torch.tensor(input_ids)
    test_masks = torch.tensor(attention_masks)

test_labels = torch.tensor(labels) 
test_identifiers = torch.tensor([int(id) for id in test_identifiers])

if 'BERT' in modelName.upper():
    test = TensorDataset(test_inputs, test_masks, test_labels, test_identifiers)
else:
    test = TensorDataset(test_labels, test_identifiers)

test_sampler = RandomSampler(test)
test_dataloader = DataLoader(test, sampler=test_sampler, batch_size=batch_size) #Test dataloader with random sampler and batched by batch size    

#Setting up optimizer
optimizer = AdamW(model.parameters(), lr = lr, eps = 2e-8)

from transformers import get_linear_schedule_with_warmup
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

#Store the average loss after each epoch so we can plot them.
loss_values = []
mcc_values = []
fone_values = []
acc_values = []

#For each epoch...
for epoch_i in range(0, epochs):
    #========================================
    #              Training
    #========================================
    #Perform one full pass over the training set.
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    #Measure how long the training epoch takes.
    t0 = time.time()

    #Reset the total loss for this epoch.
    total_loss = 0

    #Put the model into training mode. Don't be mislead--the call to 'train'just changes the *mode*, it doesn't *perform* the training.
    #'dropout' and 'batchnorm' layers behave differently during training vs. test 
    #(source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)

    model.train()
    print('\n Train:')
    #For each batch of training data...
    for step, batch in enumerate(train_dataloader):
        #Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            #Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            #Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        #Unpack this training batch from our dataloader. 
        #
        #As we unpack the batch, we'll also copy each tensor to the GPU using the 'to' method.
        #
        #'batch' contains three pytorch tensors:
        #  [0]: input ids 
        #  [1]: attention masks
        #  [2]: labels 
            
        if 'BERT' in modelName.upper():           
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            b_gender = torch.tensor([1]).to(device)
            if 'open1' in wav_location:
                wav_file = wav_location + str(audio1.alpha[batch[3].cpu().numpy()[0]]) + '.wav'
            elif 'open2' in wav_location:
                wav_file = wav_location + str(audio2.alpha[batch[3].cpu().numpy()[0]]) + '.wav'
                
        else:
            b_labels = batch[0].to(device)
            b_gender = torch.tensor([1]).to(device)
            if 'open1' in wav_location:
                wav_file = wav_location + str(audio1.alpha[batch[1].cpu().numpy()[0]]) + '.wav'
            elif 'open2' in wav_location:
                wav_file = wav_location + str(audio2.alpha[batch[1].cpu().numpy()[0]]) + '.wav'  
            
        #Always clear any previously calculated gradients before performing a backward pass.
        #PyTorch doesn't do this automatically because accumulating the gradients is "convenient while training RNNs". 
        #(source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()        

        #Perform a forward pass (evaluate the model on this training batch).
        #This will return the loss (rather than the model output) because we have provided the 'labels'.
        #The documentation for this 'model' function is here: 
        #https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification

        if 'BERT' in modelName.upper():
            outputs = model(b_input_ids,
                            b_gender,
                            wav_file,
                            token_type_ids=None,  
                            attention_mask=b_input_mask, 
                            labels=b_labels)
        else:
            outputs = model(b_gender, wav_file, labels=b_labels)


        #The call to 'model' always returns a tuple, so we need to pull the loss value out of the tuple.
        loss = outputs[0]

        #Accumulate the training loss over all of the batches so that we can calculate the average loss at the end.
        #'loss' is a Tensor containing a single value; the '.item()' function just returns the Python value from the tensor.
        total_loss += loss.item()

        #Perform a backward pass to calculate the gradients.
        loss.backward()

        #Clip the norm of the gradients to 1.0. This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        #Update parameters and take a step using the computed gradient.
        #The optimizer dictates the "update rule"--how the parameters are modified based on their gradients, the learning rate, etc.
        optimizer.step()

        #Update the learning rate.
        scheduler.step()

    #Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)            
    
    #Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

    #========================================
    #              Testing
    #========================================
    print("Testing:")
    #Prediction on test set

    #Put model in evaluation mode
    model.eval()

    #Tracking variables 
    predictions , true_labels = [], []

    #Predict 
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)

        #Unpack the inputs from our dataloader
        if 'BERT' in modelName.upper():           
            b_input_ids, b_input_mask, b_labels, b_identifier = batch
            b_gender = torch.tensor([1]).to(device)
            if 'open1' in wav_location:
                wav_file = wav_location + str(audio1.alpha[batch[3].cpu().numpy()[0]]) + '.wav'
            elif 'open2' in wav_location:
                wav_file = wav_location + str(audio2.alpha[batch[3].cpu().numpy()[0]]) + '.wav'
                
        else:
            b_labels, b_identifier = batch
            b_gender = torch.tensor([1]).to(device)
            if 'open1' in wav_location:
                wav_file = wav_location + str(audio1.alpha[batch[1].cpu().numpy()[0]]) + '.wav'
            elif 'open2' in wav_location:
                wav_file = wav_location + str(audio2.alpha[batch[1].cpu().numpy()[0]]) + '.wav'
                
        #Telling the model not to compute or store gradients, saving memory and speeding up prediction
        with torch.no_grad():
            #Forward pass, calculate logit predictions
            if 'BERT' in modelName.upper():
                outputs = model(b_input_ids, b_gender, wav_file, token_type_ids=None, attention_mask=b_input_mask)
            else:
                outputs = model(b_gender, wav_file)
        logits = outputs[0]

        #Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        #Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)
        
    print('DONE.')   
    
    #Combine the predictions for each batch into a single list of 0s and 1s.
    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

    #Combine the correct labels for each batch into a single list.
    flat_true_labels = [item for sublist in true_labels for item in sublist]
    
    #Calculate the MCC
    mcc = matthews_corrcoef(flat_true_labels, flat_predictions)
    trueProbability = []
    falseProbability = []

    for prediction in predictions:
        falseProbability.append(float(prediction[0][0]))
        trueProbability.append(float(prediction[0][1]))
    
    #Calculate metrics
    f1 = f1_score(flat_true_labels, flat_predictions)
    recall = recall_score(flat_true_labels, flat_predictions)
    precision = precision_score(flat_true_labels, flat_predictions)
    acc = accuracy_score(flat_true_labels, flat_predictions)
    auc = roc_auc_score(flat_true_labels,trueProbability)
    ba = balanced_accuracy_score(flat_true_labels,flat_predictions)
    
    #Storing calculations
    results['true_labels'] = list(map(int, flat_true_labels))
    results['predictions'] = (list(map(int, flat_predictions)))
    results['falseProbability'] = (list(map(float, falseProbability)))
    results['trueProbability'] = (list(map(float, trueProbability)))
    
    results['mcc'] = mcc
    results['f1'] = f1
    results['ba'] = ba
    results['precision'] = precision
    results['recall'] = recall
    results['acc'] = acc
    results['auc'] = auc
    results['epoch'] = epoch_i
    results['loss'] = avg_train_loss

    #Logging into database
    if logtodb:
        recordRun(configuration, short_configuration, results, tags=str(question)+","+str(modelName), db=db)
        
results['loss_values'] = loss_values
results['mcc_values'] = mcc_values