from data_utils import get_nli_dataset, get_vocab, get_embeddings_matrix, get_sequences
from keras.utils import to_categorical
from models import NLIClassifier
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from time import time
import yaml


# confirm TensorFlow sees the GPU
from tensorflow.python.client import device_lib
assert 'GPU' in str(device_lib.list_local_devices())

# confirm Keras sees the GPU
from keras import backend
assert len(backend.tensorflow_backend._get_available_gpus()) > 0


__author__ = 'avijitv'


def make_plots(history, path, title, epochs):
    hist = pd.DataFrame(history.history)
    plt.style.use("ggplot")
    plt.figure(figsize=(12, 12))
    plt.suptitle(title)
    plt.subplot(2, 1, 1)
    plt.plot([i for i in range(1, epochs + 1)], hist["loss"], color='blue')
    plt.plot([i for i in range(1, epochs + 1)], hist["val_loss"], color='red')
    plt.ylabel('Categorical Cross-entropy Loss')

    plt.subplot(2, 1, 2)
    plt.plot([i for i in range(1, epochs + 1)], hist["acc"], color='blue')
    plt.plot([i for i in range(1, epochs + 1)], hist["val_acc"], color='red')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')

    plt.savefig(path)


def main():
    with open('config.yaml', 'r') as infile:
        config = yaml.load(infile)

    dataset = get_nli_dataset(path=config['data']['nli_data_dir'])
    all_sentences = []
    for split in ['train', 'test', 'dev']:
        all_sentences += dataset[split]['s1'] + dataset[split]['s2']
    max_seq_len = max(len(s) for s in all_sentences)
    print('Max sentence length: {0}'.format(max_seq_len))
    vocab = get_vocab(all_sentences)
    word_embedding_matrix = get_embeddings_matrix(embeddings_path=config['data']['embeddings_path'],
                                                  vocab=vocab)
    print('Embedding Matrix Shape: {0}'.format(word_embedding_matrix.shape))

    try:
        num_train_samples = int(config['data']['num_train_samples'])
    except ValueError:
        num_train_samples = len(dataset['train']['s1']) + 1
    try:
        num_dev_samples = int(config['data']['num_dev_samples'])
    except ValueError:
        num_dev_samples = len(dataset['dev']['s1']) + 1
    try:
        num_test_samples = int(config['data']['num_test_samples'])
    except ValueError:
        num_test_samples = len(dataset['test']['s1']) + 1

    s1_train_x = get_sequences(sentences=dataset['train']['s1'], max_seq_len=max_seq_len, vocab=vocab)[:num_train_samples]
    s2_train_x = get_sequences(sentences=dataset['train']['s2'], max_seq_len=max_seq_len, vocab=vocab)[:num_train_samples]
    s1_dev_x = get_sequences(sentences=dataset['dev']['s1'], max_seq_len=max_seq_len, vocab=vocab)[:num_dev_samples]
    s2_dev_x = get_sequences(sentences=dataset['dev']['s2'], max_seq_len=max_seq_len, vocab=vocab)[:num_dev_samples]
    s1_test_x = get_sequences(sentences=dataset['test']['s1'], max_seq_len=max_seq_len, vocab=vocab)[:num_test_samples]
    s2_test_x = get_sequences(sentences=dataset['test']['s2'], max_seq_len=max_seq_len, vocab=vocab)[:num_test_samples]

    print('Train Shape: S1: {0}, S2: {1}'.format(s1_train_x.shape, s2_train_x.shape))
    print('Dev Shape: S1: {0}, S2: {1}'.format(s1_dev_x.shape, s2_dev_x.shape))
    print('Test Shape: S1: {0}, S2: {1}'.format(s1_test_x.shape, s2_test_x.shape))

    train_y = to_categorical(dataset['train']['target'], num_classes=3)[:num_train_samples]
    dev_y = to_categorical(dataset['dev']['target'], num_classes=3)[:num_dev_samples]
    test_y = to_categorical(dataset['test']['target'], num_classes=3)[:num_test_samples]

    print('Y Train Shape: {0}'.format(train_y.shape))
    print('Y Dev Shape: {0}'.format(dev_y.shape))
    print('Y Test Shape: {0}'.format(test_y.shape))

    if not os.path.exists(config['model']['path']):
        os.makedirs(config['model']['path'])

    nli_classifier = NLIClassifier(config=config['model'],
                                   vocab_size=len(vocab),
                                   embedding_matrix=word_embedding_matrix,
                                   max_seq_len=max_seq_len)
    fit_start_time = time()
    history = nli_classifier.fit(s1_train_x=s1_train_x,
                                 s2_train_x=s2_train_x,
                                 s1_dev_x=s1_dev_x,
                                 s2_dev_x=s2_dev_x,
                                 train_y=train_y,
                                 dev_y=dev_y)
    fit_end_time = time()
    fit_time = fit_end_time - fit_start_time
    print('Fit time: {0}'.format(round(fit_time, 3)))

    make_plots(history=history,
               path='{0}/NLITraining_{1}.png'.format(config['model']['path'],
                                                     config['model']['encoder']['type']),
               title='NLI Training - {0}'.format(config['model']['encoder']['type']),
               epochs=config['model']['training']['epochs'])

    pred_y = nli_classifier.predict(s1_x=s1_test_x,
                                    s2_x=s2_test_x)

    test_y = np.argmax(test_y, axis=1)
    pred_y = np.argmax(pred_y, axis=1)
    print()
    print('Accuracy - {0}'.format(round(accuracy_score(test_y, pred_y), 3)))
    print('Classification Report - ')
    print(classification_report(y_true=test_y, y_pred=pred_y))


    encode_start_time = time()
    encoded_s1 = nli_classifier.encode(s1_x=s1_dev_x)
    encode_end_time = time()
    print('Encoded S1 shape: {0}'.format(encoded_s1.shape))
    encoding_time = encode_end_time - encode_start_time
    print('Encoding time for Dev Set: {0}'.format(round(encoding_time, 3)))
    print('Encoding time per sample: {0}'.format(round(encoding_time / len(s1_dev_x), 3)))

if __name__ == '__main__':
    main()
