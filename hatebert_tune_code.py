import csv
import datetime
import logging
import os
import random
import re
import time
from datetime import datetime as dt

import emoji
import numpy as np
import pandas as pd
import torch
from keras_preprocessing.sequence import pad_sequences
# from _datetime import datetime as dt
from sklearn.metrics import classification_report
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler, TensorDataset)
from tqdm.notebook import trange, tqdm
from transformers import (WEIGHTS_NAME, AdamW, AutoModel, AutoModelWithLMHead,
                          AutoTokenizer, BertConfig, BertForMaskedLM,
                          BertForSequenceClassification, BertTokenizer,
                          CamembertConfig, CamembertForMaskedLM,
                          CamembertTokenizer, DistilBertConfig,
                          DistilBertForMaskedLM, DistilBertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel,
                          OpenAIGPTTokenizer, PreTrainedModel,
                          PreTrainedTokenizer, RobertaConfig,
                          RobertaForMaskedLM, RobertaTokenizer,
                          get_linear_schedule_with_warmup)

# Logger stuff
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
logger = logging.getLogger(__name__)



def load_train_test_data(tokenizer):

    '''
    Function to load training and test data. The input format is one tweet per line:
    tweet_id \t tweet text \t OFF/NOT
    :param language: the tweets language, it is used just for paths, can be removed
    :param tokenizer: BERT tokenizer, output of the training code
    :return: the list of
        train_input_ids, train_labels, train_attention_masks,test_input_ids, test_labels, test_attention_masks
        which stand for tokenized tweet texts, labels and computed attention mask for training and test data respectively
    '''
    train_file_path = '/content/drive/MyDrive/reprod_code/data/full_train.csv'
    test_file_path = '/content/drive/MyDrive/reprod_code/data/full_test.csv'

    # List of all tweets text
    train_tweets = []
    # List of all labels
    train_labels = []

    # List of all tweets text
    test_tweets = []
    # List of all labels
    test_labels = []

    # -----------------------------------------------------------------
    # Parse Training Set
    with open(train_file_path, encoding='utf-8') as input_file:
        # For each tweet
        count = 0
        for line in csv.reader(input_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_ALL):
            if line[0] != 'id' and len(line) == 3:
                full_line = line[1]
                full_line = re.sub(r'#([^ ]*)', r'\1', full_line)
                full_line = re.sub(r'https.*[^ ]', 'URL', full_line)
                full_line = re.sub(r'http.*[^ ]', 'URL', full_line)
                full_line = emoji.demojize(full_line)
                full_line = re.sub(r'(:.*?:)', r' \1 ', full_line)
                full_line = re.sub(' +', ' ', full_line)

                # Binary prediction

                if line[2] == 'OFF':
                    label = 1
                else:
                    label = 0

                # Save tweet's text and label
                train_tweets.append(full_line)
                train_labels.append(label)

    # -----------------------------------------------------------------
    # Parse Test Set
    with open(test_file_path, encoding='utf-8') as input_file:
        # For each tweet
        for line in csv.reader(input_file, delimiter=","):
            if line[0] != 'id' and len(line) == 3:
                full_line = line[1]
                full_line = re.sub(r'#([^ ]*)', r'\1', full_line)
                full_line = re.sub(r'https.*[^ ]', 'URL', full_line)
                full_line = re.sub(r'http.*[^ ]', 'URL', full_line)
                full_line = emoji.demojize(full_line)
                full_line = re.sub(r'(:.*?:)', r' \1 ', full_line)
                full_line = re.sub(' +', ' ', full_line)

                # Binary prediction

                if line[2] == 'OFF':
                    label = 1
                else:
                    label = 0



                # Save tweet's text and label
                test_tweets.append(full_line)
                test_labels.append(label)

    # List of all tokenized tweets
    train_input_ids = []
    test_input_ids = []

    # For every tweet in the training set
    for sent in train_tweets:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = tokenizer.encode(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=100,  # orignal value 512 - change Tommaso

            # This function also supports truncation and conversion
            # to pytorch tensors, but we need to do padding, so we
            # can't use these features :( .
            # max_length = 128,          # Truncate all sentences.
            # return_tensors = 'pt',     # Return pytorch tensors.
        )

        # Add the encoded tweet to the list.
        train_input_ids.append(encoded_sent)

    # For every tweet in the test set
    for sent in test_tweets:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = tokenizer.encode(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=100, # orignal value 512 - change Tommaso

            # This function also supports truncation and conversion
            # to pytorch tensors, but we need to do padding, so we
            # can't use these features :( .
            # max_length = 128,          # Truncate all sentences.
            # return_tensors = 'pt',     # Return pytorch tensors.
        )

        # Add the encoded tweet to the list.
        test_input_ids.append(encoded_sent)

    # # Pad our input tokens with value 0.
    # # "post" indicates that we want to pad and truncate at the end of the sequence,
    # # as opposed to the beginning.

    train_input_ids = pad_sequences(train_input_ids, maxlen=100, dtype="long",
                          value=tokenizer.pad_token_id, truncating="pre", padding="pre")

    test_input_ids = pad_sequences(test_input_ids, maxlen=100, dtype="long",
                                    value=tokenizer.pad_token_id, truncating="pre", padding="pre")



    # Create attention masks
    # The attention mask simply makes it explicit which tokens are actual words versus which are padding
    train_attention_masks = []
    test_attention_masks = []

    # For each tweet in the training set
    for sent in train_input_ids:
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for this sentence.
        train_attention_masks.append(att_mask)

    # For each tweet in the test set
    for sent in test_input_ids:
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for this sentence.
        test_attention_masks.append(att_mask)

    # Return the list of encoded tweets, the list of labels and the list of attention masks
    return train_input_ids, train_labels, train_attention_masks, test_input_ids, test_labels, test_attention_masks


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# ======================================================================================================================
# Part of the code comes from https://mccormickml.com/2019/07/22/BERT-fine-tuning/
# ======================================================================================================================
# ---------------------------- Main ----------------------------

# Input language, is used just for paths
#language = 'tr'

# !! the code was inserted into a function to load it into a jupyter notbook in Google Colab !!
# !! No alteration were made to the code's functioning
def run_hatebert(seed_val, numb_runs):
	
  f_scores = []
  seeds = []
  runs=[]
  epochs_num=[]

  # !! This loop was added to test for variance within one seed value !!
  for run_num in range(numb_runs):
		# Directory where the pre-trained model can be found (after pre-traing from Huggingface)
    model_dir = '/content/drive/MyDrive/reprod_code/HateBERT_offenseval'


    # Returns a datetime object containing the local date and time (used for output_model_dir)
    dateTimeObj = 'trial' #' str(dt.now()).replace(" ", "_")

    # Directory in which the model will be saved along with the log
    output_model_dir = '/content/drive/MyDrive/reprod_code'  + dateTimeObj + "/" # "../out/" + language + '/models/' + dateTimeObj + "/"
		#r'C:\Users\roric\Master HLT\Period 6\NLP Experiments\reprod_code'  + dateTimeObj + "\\" 
		# Make dir for model serializations
    os.makedirs(os.path.dirname(output_model_dir), exist_ok=True)

		# Log stuff: print logger on file in output_model_dir/log.log
    logging.basicConfig(filename=output_model_dir + 'log.log', level=logging.DEBUG)

		# Log stuff: print logger also on stderr
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    # -----------------------------
    # Load Pre-trained BERT model
    # -----------------------------
    config_class, model_class, tokenizer_class = (BertConfig, BertForSequenceClassification, BertTokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load a trained model and vocabulary pre-trained for specific language
    # logger.info("Loading model") #from: '" + model_dir + "', it may take a while...")

    # Load pre-trained Tokenizer from directory, change this to load a tokenizer from ber package
    tokenizer = tokenizer_class.from_pretrained(model_dir)

    # Load Bert for classification 'container'
    model = BertForSequenceClassification.from_pretrained(
      model_dir, # Use pre-trained model from its directory, change this to use a pre-trained model from bert
      num_labels = 2, # The number of output labels--2 for binary classification.
              # You can increase this for multi-class tasks.
      output_attentions = False, # Whether the model returns attentions weights.
      output_hidden_states = False, # Whether the model returns all hidden-states.
    )

    # Set the model to work on CPU if no GPU is present
    model.to(device)
    # logger.info("HateBERT for classification model has been loaded!")
    # --------------------------------------------------------------------
    # -------------------------- Load test data --------------------------
    # --------------------------------------------------------------------

    # The loading eval data return:
    # - input_ids:         the list of all tweets already tokenized and ready for bert (with [CLS] and [SEP])
    # - labels:            the list of labels, the i-th index corresponds to the i-th position in input_ids
    # - attention_masks:   a list of [0,1] for every input_id that represent which token is a padding token and which is not

    train_inputs, train_labels, train_masks, validation_inputs, validation_labels, validation_masks = load_train_test_data(tokenizer)
    # --------------------------------------------------------------------
    # -------------------- Split train and validation --------------------
    # --------------------------------------------------------------------
    # Convert all inputs and labels into torch tensors, the required datatype for our model.

    # Tweets
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)

    # Labels
    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)

    # Attention masks
    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)

    # We will use a DataLoader, it helps save on memory during training because, unlike a for loop, with an iterator
    # the entire dataset does not need to be loaded into memory
    # The DataLoader needs to know our batch size for training, so we specify it here.
    # For fine-tuning BERT on a specific task, the authors recommend a batch size of
    # 16 or 32.

    batch_size = 32

    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create the DataLoader for our validation set.
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    # Note that the number of batch has to be the same, this means that we have to aggregate results in the end
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    # --------------------------------------------------------------------
    # -------------- Optimizer and Learning Rate Scheduler ---------------
    # --------------------------------------------------------------------
    # For the purposes of fine-tuning, the authors recommend choosing from the following values:
    # Learning rate (Adam): 5e-5, 3e-5, 2e-5 (We’ll use 2e-5).
    # Number of epochs: 2, 3, 4 (We’ll use 4).
    #
    #
    # Note: AdamW is a class from the HuggingFace library (as opposed to PyTorch)
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
              lr=1e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
              eps=1e-8  # args.adam_epsilon  - default is 1e-8.
            )

    # Number of training epochs (authors recommend between 2 and 4)
    epochs = 5

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                          num_warmup_steps=0,  # Default value in run_glue.py
                          num_training_steps=total_steps)

    # --------------------------------------------------------------------
    # Now we are ready to prepare and run the training/evaluation
    # --------------------------------------------------------------------
    #
    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128


    # Set the seed value all over the place to make this reproducible.
    # seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # Store the average loss after each epoch so we can plot them.
    loss_values = []

    # For each epoch...
    for epoch_i in tqdm(range(0, epochs), desc="Training"):

      # ========================================
      #               Training
      # ========================================

      # Store true lables for global eval
      gold_labels = []
      # Store  predicted labels for global eval
      predicted_labels = []

      # Perform one full pass over the training set.

      # logger.info("")
      # logger.info('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
      # logger.info('Training...')

      # Measure how long the training epoch takes.
      t0 = time.time()

      # Reset the total loss for this epoch.
      total_loss = 0

      # Put the model into training mode. Don't be mislead--the call to
      # `train` just changes the *mode*, it doesn't *perform* the training.
      # `dropout` and `batchnorm` layers behave differently during training
      # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
      model.train()

      # For each batch of training data...
      # the tqdm instruction mess with prints on terminal but it can be useful to understand what is the current
      # batch at any time
      for step, batch in tqdm(enumerate(train_dataloader), desc="Batch"):

        # # Progress update every 40 batches.
        # if step % 40 == 0 and not step == 0:
        #     # Calculate elapsed time in minutes.
        #     elapsed = format_time(time.time() - t0)

        #     # Report progress.
        #     logger.info('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader.
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because
        # accumulating the gradients is "convenient while training RNNs".
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()

        # Perform a forward pass (evaluate the model on this training batch).
        # This will return the loss (rather than the model output) because we
        # have provided the `labels`.
        # The documentation for this `model` function is here:
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        outputs = model(b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels)

        # The call to `model` always returns a tuple, so we need to pull the
        # loss value out of the tuple.
        loss = outputs[0]

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value
        # from the tensor.
        total_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

      # Calculate the average loss over the training data.
      avg_train_loss = total_loss / len(train_dataloader)

      # Store the loss value for plotting the learning curve.
      loss_values.append(avg_train_loss)

      # logger.info("")
      # logger.info("  Average training loss: {0:.2f}".format(avg_train_loss))
      # logger.info("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

      # ------------------------------------------------------------------------------------------------------------------
      # Todo: Cut code from here to remove the validation step: the loading function has to be changed in order
      #  to parse the training set only
      # ------------------------------------------------------------------------------------------------------------------

      # ========================================
      #               Validation
      # ========================================
      # After the completion of each training epoch, measure our performance on
      # our validation set.

      # logger.info("")
      # logger.info("Running Validation...")

      t0 = time.time()

      # Put the model in evaluation mode--the dropout layers behave differently
      # during evaluation.
      model.eval()

      # Tracking variables
      eval_loss, eval_accuracy = 0, 0
      nb_eval_steps, nb_eval_examples = 0, 0

      # Evaluate data for one epoch
      for batch in validation_dataloader:
        # Add batch to GPU/CPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():
          # Forward pass, calculate logit predictions.
          # This will return the logits rather than the loss because we have
          # not provided labels.
          # token_type_ids is the same as the "segment ids", which
          # differentiates sentence 1 and 2 in 2-sentence tasks.
          # The documentation for this `model` function is here:
          # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
          outputs = model(b_input_ids,
                  token_type_ids=None,
                  attention_mask=b_input_mask)

        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        # Accumulate the total accuracy.
        eval_accuracy += tmp_eval_accuracy

        # Track the number of batches
        nb_eval_steps += 1

        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()

        # Store gold labels single list
        gold_labels.extend(labels_flat)
        # Store predicted labels single list
        predicted_labels.extend(pred_flat)

        # The classification report is printed on the log, note that print one report for each validation epoch,
        # if we want to compute an average P/R/F1 we can do the same as accuracy, that is an accumulator that
        # stores P/R over epochs or compute the average at the end

        # logger.info(classification_report(labels_flat,pred_flat, digits=4))

      # ------------------------------------------------------------------------------------------------------------------
      # Todo: Cut code until here to remove the validation step
      # ------------------------------------------------------------------------------------------------------------------

      #     # Report the final accuracy for this validation run.
      #     logger.info("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
      #     logger.info("  Validation took: {:}".format(format_time(time.time() - t0)))


      #     logger.info("")
      #     logger.info("Evaluation on full prediction per epoch!")
      #     logger.info("Gold labels" + str(len(gold_labels)))
      #     logger.info("Predicted labels" + str(len(predicted_labels)))
      metrics = pd.DataFrame(classification_report(gold_labels,predicted_labels, digits=4, output_dict=True)).T
      f_scores.append(metrics['f1-score']['macro avg'])
      seeds.append(seed_val)
      runs.append(run_num)
      epochs_num.append(epoch_i)

    # !! The indentation causes the looping to be cancelled, each seed value is only used once !!
    return f_scores, runs, epochs_num