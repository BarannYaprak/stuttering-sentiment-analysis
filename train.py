import pandas as pd
import numpy as np
import re
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader
from transformers import AdamW, get_cosine_with_hard_restarts_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
device = torch.device("cuda")

def get_dataset():
  df = pd.read_json(r"/content/drive/MyDrive/ytu_donem/1-sentetik_karma_veriler.json")
  text = df["SENTENCE"].values.tolist()
  label = df['LABEl'].values.tolist()
  return text, label

def CleanTxt(text):
  text = re.sub(r'@[A-Za-z0-9]+', "", text)
  text = re.sub(r'bit.ly/\s+', "", text)
  text = re.sub(r" (\A\s+|\s+\Z)", "", text)
  text = re.sub(r"https?:\/\/\s+", "", text)
  text = re.sub(r" \b[a-za-z]\b","", text)
  text = re.sub(r'(.)\1+', r"\1\1", text)
  text = text.replace("I","1").lower()
  return text
text, label = get_dataset()
cleaned_text = [CleanTxt(tweet) for tweet in text]

def get_input_parameters(tokenizer_name,x_train,y_train,batch_size):
  tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

  input_ids_train = []
  attention_masks_train = []

  for text in x_train:
    encoded_dict_train = tokenizer.encode_plus(
                                              text,
                                              add_special_tokens = True,
                                              pad_to_max_length = True,
                                              max_length = 128,
                                              return_attention_mask = True,
                                              return_tensors = "pt")
    input_ids_train.append(encoded_dict_train["input_ids"])
    attention_masks_train.append(encoded_dict_train["attention_mask"])

  input_ids_train = torch.cat(input_ids_train, dim = 0)
  attention_masks_train = torch.cat(attention_masks_train, dim = 0)
  labels_train = torch.tensor(y_train)



  dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)


  train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

  return train_dataloader

def GetModel(model_name,learning_rate,epochs,train_dataloader):
  model = BertForSequenceClassification.from_pretrained(
  model_name,
  num_labels= 4,
  output_attentions = True,
  output_hidden_states = False
  )

  model.to(device)

  optimizer = AdamW(model.parameters(), lr = learning_rate)
  total_steps = len(train_dataloader) * epochs
  scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                        num_warmup_steps = 0,
                                                        num_cycles = epochs,
                                                        num_training_steps = total_steps)
  return model,optimizer,scheduler

def Train_Bert(model,optimizer,scheduler,train_dataloader,epochs):
  for epoch_i in range(0,epochs):
    model.train()
    total_train_loss = 0
    for step, batch in enumerate(train_dataloader):
      b_input_ids = batch[0].to(device)
      b_input_mask = batch[1].to(device)
      b_labels = batch[2].to(device)
      model.zero_grad()

      outputs = model(b_input_ids,
                      token_type_ids = None,
                      attention_mask = b_input_mask,
                      labels = b_labels)

      loss = outputs.loss

      total_train_loss += loss.item()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      scheduler.step()

  avg_train_loss = total_train_loss / len(train_dataloader)
  print(f"Epoch {epoch_i+ 1} / {epochs} - Average training loss: {avg_train_loss}")

  return model

X_train, X_test, y_train, y_test = train_test_split(cleaned_text, label, test_size=0.2, random_state=42, shuffle= True)

train_dataloader = get_input_parameters("ytu-ce-cosmos/turkish-base-bert-uncased",X_train,y_train,32)

model, optimizer, scheduler = GetModel("ytu-ce-cosmos/turkish-base-bert-uncased",2e-5,12,train_dataloader)

model = Train_Bert(model,optimizer,scheduler,train_dataloader,12)

model.save_pretrained("")

test_data = get_input_parameters("ytu-ce-cosmos/turkish-base-bert-uncased",X_test,y_test,32)

def test_model(model, test_data, device):
    model.eval()
    all_preds = []
    all_labels = []

    for batch in test_data:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)
            logits = outputs.logits

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().detach().numpy())
        all_labels.extend(b_labels.cpu().detach().numpy())

    return all_preds, all_labels

all_preds, all_labels = test_model(model, test_data, device)
print(f"Predictions: {all_preds}")
print(f"True Labels: {all_labels}")

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')
accuracy = accuracy_score(all_labels, all_preds)


print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
print(f"Accuracy: {accuracy}")