from transformers import BertModel, BertPreTrainedModel
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch
from torchvggish2.vggish import VGGish
import torchaudio

class BertForSequenceClassificationWithVGG(BertPreTrainedModel):
    def __init__(self, config,num_gender_features,numVectors,useLSTM,useLastVectors):
        super().__init__(config)

         #If the Gender features given to model
        if(num_gender_features == 0):
            self.useGender = False
        else:
            self.useGender = True  
        
        #If using last vectors for model
        if useLastVectors:
            self.useLastVectors = True
        else:
            self.useLastVectors = False    
        self.numVectors = numVectors
        
        
        self.input_dim = 128 #Input dimensions

        if useLSTM:
            #If using LSTM
            print("using LSTM")
            self.useLSTM = True
            self.hidden_dim = 128
            self.n_layers = 1     
            self.lstm_layer = nn.LSTM(self.input_dim, self.hidden_dim, self.n_layers, batch_first=True)
            self.numVGGFeatures = self.hidden_dim
        else:
            self.useLSTM = False
            self.numVGGFeatures = self.input_dim * numVectors
            
        self.num_labels = config.num_labels
        self.vggish_layer = VGGish() #Loading pretrained VGGish
        self.bert = BertModel(config) #Loading pretrained BERT
        self.dropout = nn.Dropout(config.hidden_dropout_prob) #Dropout layer
        
        self.classifier = nn.Linear(config.hidden_size+self.numVGGFeatures+num_gender_features, self.config.num_labels) #Classifier layer

        self.init_weights() #Initializing weights

    def forward(self, input_ids, gender_ids, wav_file, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, fs=None):
        #BERT output
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        #VGGish output
        audio_outputs = self.vggish_layer(wav_file)

        pooled_output = outputs[1]
        
        if self.useLSTM:
            #If using LSTM, pass VGGIsh output through LSTM
            audio_outputs = audio_outputs.unsqueeze(0)
            lstm_out, (ht, ct) = self.lstm_layer(torch.tensor(audio_outputs))
            audio_outputs = ht[-1]
            audio_pooled_output = self.dropout(audio_outputs)
        else:
            if self.useLastVectors:
                #If using last vectors                
                maxVectors = audio_outputs.size()[0] - 1 
                initial = maxVectors
                vectorRange = range(maxVectors-1,maxVectors-self.numVectors,-1)
            else:
                initial = 0
                vectorRange = range(1, self.numVectors,1)
            audio_pooled_output = torch.tensor([audio_outputs[initial].tolist()]).cuda()
            
            for wavVector in vectorRange:
                audioTempOutput = torch.tensor([audio_outputs[wavVector].tolist()]).cuda()
                audio_pooled_output = torch.cat((audio_pooled_output,audioTempOutput),1)            
        
        

        pooled_output = torch.cat((pooled_output,audio_pooled_output),1) #Pooling both BERT and VGGishLSTM outputs
        pooled_output = self.dropout(pooled_output)

        #Calculating the logits
        if self.useGender:
            logits = self.classifier(torch.cat((pooled_output,gender_ids.view(-1, 1)),1))
        else:
            logits = self.classifier(pooled_output)
       
        outputs = (logits,) + outputs[2:]  #Add hidden states and attention if they are here

        #Calculating the loss
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  #(loss), logits, (hidden_states), (attentions)    
