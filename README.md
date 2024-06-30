# HMM-POS-Tagger

This project implements a Hidden Markov Model (HMM) for part-of-speech tagging using the Wall Street Journal section of the Penn Treebank.

## Introduction

This project involves creating an HMM for part-of-speech tagging. The tasks include vocabulary creation, model learning, greedy decoding, and Viterbi decoding. The data is divided into training, development, and test sets. The training and development sets contain sentences with human-annotated part-of-speech tags, while the test set contains raw sentences for which we predict the tags.

## Project Structure

- `data/`
  - `train`: Training data with part-of-speech tags.
  - `dev`: Development data with part-of-speech tags.
  - `test`: Test data with raw sentences.
- `vocab.txt`: The vocabulary file created from the training data.
- `hmm.json`: The learned HMM model containing emission and transition probabilities.
- `greedy.out`: Predictions on the test data using the greedy decoding algorithm.
- `viterbi.out`: Predictions on the test data using the Viterbi decoding algorithm.
- `eval.py`: Evaluation script to assess the model's accuracy.
- `main.py`: Main script to run the tasks.
- `README.md`: This file.

## Tasks

### Task 1: Vocabulary Creation

- Created a vocabulary using the training data.
- Handled unknown words by replacing rare words (occurrences < 2) with a special token `<unk>`.
- Output the vocabulary to `vocab.txt` in the format: `word type \t index \t occurrences`, sorted by occurrences in descending order.
- The first line is the special token `<unk>`.
- **Results**:
  - Threshold for rare words: 2
  - Total vocabulary size: 23,182
  - Total occurrences of `<unk>` token: 20,011

### Task 2: Model Learning

- Learned an HMM from the training data.
- Calculated emission and transition parameters.
- Saved the model to `hmm.json`

- **Results**:
 - Number of transition parameters: 1,392
 - Number of emission parameters: 30,303

### Task 3: Greedy Decoding with HMM
- Implemented the greedy decoding algorithm.
- Evaluated the model on the development data.
- Predicted part-of-speech tags for the test data and saved the results to greedy.out.
- **Results**:
 - Accuracy on dev data: 93.50%

### Task 4: Viterbi Decoding with HMM
- Implemented the Viterbi decoding algorithm.
- Evaluated the model on the development data.
- Predicted part-of-speech tags for the test data and saved the results to viterbi.out.
- **Results**:
 - Accuracy on dev data: 94.77%

### Usage

To reproduce the results and run the code:

Ensure the data files (train, dev, test) are in the data/ directory.
Run the main script to perform all tasks:
```
python main.py
```

### Evaluation

To evaluate the predictions:

Use the eval.py script with the predicted file and the gold-standard file:
```
python eval.py -p greedy.out -g data/dev
python eval.py -p viterbi.out -g data/dev
```
