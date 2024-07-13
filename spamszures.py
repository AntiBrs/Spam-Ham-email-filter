import os
import re
import numpy as np


def preprocess_data(directory, stopwords_file, filenames):
    files = []
    contents = {}
    names = []
    stopwords = read_stopwords(stopwords_file)
    with open(filenames, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if os.path.exists(os.path.join(directory, line.rstrip('\n'))):
                names.append(line.rstrip('\n'))
    for filename in names:
        contents = {}
        with open(os.path.join(directory, filename), 'r', encoding='latin-1') as file:
            content = file.read()
            content = content.lower()
            content = re.sub(r'\W', ' ', content)
            content = re.sub(r'Subject: ', '', content)
            words = content.split()
            filtered_words = [word for word in words if word not in stopwords]
            for word in filtered_words:
                contents[word] = contents.get(word, 0) + 1
        files.append(contents)

    return files


def read_stopwords(filename):
    with open(filename, 'r') as file:
        stopwords = file.readlines()
        stopwords = [word.strip() for word in stopwords]
    return stopwords


def predict(email, p_spam, p_ham, vocabulary):
    r = np.log(p_spam) - np.log(p_ham)
    for word in email:
        if word in vocabulary:
            r += email[word] * (np.log(vocabulary[word][0]) - np.log(vocabulary[word][1]))
    return r > 1


def test_naive_bayes(test_ham, test_spam, p_spam, p_ham, vocabulary):
    ham_correct = sum([not predict(email, p_spam, p_ham, vocabulary) for email in test_ham])
    spam_correct = sum([predict(email, p_spam, p_ham, vocabulary) for email in test_spam])
    total_ham = len(test_ham)
    total_spam = len(test_spam)
    error = (ham_correct + spam_correct) / (total_ham + total_spam)
    return error


train_ham = preprocess_data('enron6/ham', 'stopwords.txt', 'train.txt')
train_spam = preprocess_data('enron6/spam', 'stopwords.txt', 'train.txt')

test_ham = preprocess_data('enron6/ham', 'stopwords.txt', 'test.txt')
test_spam = preprocess_data('enron6/spam', 'stopwords.txt', 'test.txt')

p_spam = len(train_spam) / (len(train_spam) + len(train_ham))
p_ham = len(train_ham) / (len(train_spam) + len(train_ham))

alphas = [0, 0.01, 0.1, 1]
for alpha in alphas:
    vocabulary = {}
    for email in train_ham:
        for word in email:
            if word not in vocabulary:
                vocabulary[word] = [0, 0]

    for email in train_spam:
        for word in email:
            if word not in vocabulary:
                vocabulary[word] = [0, 0]

    sum_spam = 0
    sum_ham = 0
    lb = 0.000000000001
    for word in vocabulary:
        sum_spam += max(lb, sum(email.get(word, 0) for email in train_spam))
        sum_ham += max(lb, sum(email.get(word, 0) for email in train_ham))

    for word in vocabulary:
        spam_count = max(lb, sum(email.get(word, 0) for email in train_spam))
        ham_count = max(lb, sum(email.get(word, 0) for email in train_ham))
        total_word_count = spam_count + ham_count
        vocabulary[word] = [(spam_count + alpha) / (sum_spam + alpha * len(vocabulary)),
                            (ham_count + alpha) / (sum_ham + alpha * len(vocabulary))]

    if alpha == 0:
        print(f"\nWithout additive smoothing:")
        error = test_naive_bayes(train_ham, train_spam, p_spam, p_ham, vocabulary)
        print("Training error:", 1 - error)

        error = test_naive_bayes(test_ham, test_spam, p_spam, p_ham, vocabulary)
        print("Test error:", 1 - error)
    else:
        print(f"\nWith additive smoothing (alpha = {alpha}):")
        error = test_naive_bayes(train_ham, train_spam, p_spam, p_ham, vocabulary)
        print("Training error:", 1 - error)

        error = test_naive_bayes(test_ham, test_spam, p_spam, p_ham, vocabulary)
        print("Test error:", 1 - error)