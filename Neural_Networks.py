import pandas as pd
import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings('ignore')

yTrain = pd.read_csv('yTrain.csv').values
yTest = pd.read_csv('yTest.csv').values
yTrain = np.asarray(yTrain).astype('float32').ravel()
yTest = np.asarray(yTest).astype('float32').ravel()

def handle_data(dataset):
    xTrain = pd.read_csv(f'{dataset}_df_train.csv').values
    xTest = pd.read_csv(f'{dataset}_df_test.csv').values
    xTrain = np.asarray(xTrain).astype('float32')
    xTest = np.asarray(xTest).astype('float32')
    return xTrain, xTest

def Dense(xTrain, xTest, yTrain, yTest, dataset, shape):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(shape,)))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
    model.fit(xTrain, yTrain, epochs=5, batch_size=2056, callbacks=[early_stopping], validation_split=0.1)
    loss, accuracy = model.evaluate(xTest, yTest)
    print(f"Dense Neural Networks test accuracy for {dataset} dataset: {accuracy*100}%\n")

def CNN(xTrain, xTest, yTrain, yTest, dataset, shape):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=int(np.max(xTrain))+1, output_dim=64, input_length=shape))
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.GlobalMaxPooling1D())
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    model.fit(xTrain, yTrain, epochs=5, batch_size=2056, validation_split=0.1, callbacks=[early_stopping])
    loss, accuracy = model.evaluate(xTest, yTest)
    print(f"Convolutional Neutral Networks (CNN) test accuracy for {dataset} dataset: {accuracy*100}%\n")

def RNN(xTrain, xTest, yTrain, yTest, dataset, shape):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.SimpleRNN(30, return_sequences=True, input_shape=(shape,1)))
    model.add(tf.keras.layers.SimpleRNN(30))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
    model.fit(xTrain, yTrain, epochs=5, batch_size=2056, callbacks=[early_stopping], validation_split=0.1)
    loss, accuracy = model.evaluate(xTest, yTest)
    print(f"Recurrent Neutral Networks (RNN) test accuracy for {dataset} dataset: {accuracy*100}%\n")


# Binary Dataset
binary_df_train, binary_df_test = handle_data('binary')
Dense(binary_df_train, binary_df_test, yTrain, yTest, 'binary', binary_df_train.shape[1])
CNN(binary_df_train, binary_df_test, yTrain, yTest, 'binary', binary_df_train.shape[1])
RNN(binary_df_train, binary_df_test, yTrain, yTest, 'binary', binary_df_train.shape[1])

# Bag of Words Dataset
bow_df_train, bow_df_test = handle_data('bow')
Dense(bow_df_train, bow_df_test, yTrain, yTest, 'bow', bow_df_train.shape[1])
CNN(bow_df_train, bow_df_test, yTrain, yTest, 'bow', bow_df_train.shape[1])
RNN(bow_df_train, bow_df_test, yTrain, yTest, 'bow', bow_df_train.shape[1])

# Tfidf Dataset
tfidf_df_train, tfidf_df_test = handle_data('tfidf')
Dense(tfidf_df_train, tfidf_df_test, yTrain, yTest, 'tfidf', tfidf_df_train.shape[1])
CNN(tfidf_df_train, tfidf_df_test, yTrain, yTest, 'tfidf', tfidf_df_train.shape[1])
RNN(tfidf_df_train, tfidf_df_test, yTrain, yTest, 'tfidf', tfidf_df_train.shape[1])

# Hashing Dataset
hash_df_train, hash_df_test = handle_data('hash')
Dense(hash_df_train, hash_df_test, yTrain, yTest, 'hashing', hash_df_train.shape[1])
RNN(hash_df_train, hash_df_test, yTrain, yTest, 'hashing', hash_df_train.shape[1])

# LDA Dataset
lda_df_train, lda_df_test = handle_data('lda')
Dense(lda_df_train, lda_df_test, yTrain, yTest, 'LDA', lda_df_train.shape[1])
CNN(lda_df_train, lda_df_test, yTrain, yTest, 'LDA', lda_df_train.shape[1])
RNN(lda_df_train, lda_df_test, yTrain, yTest, 'LDA', lda_df_train.shape[1])

# PCA Dataset
pca_df_train, pca_df_test = handle_data('pca')
Dense(pca_df_train, pca_df_test, yTrain, yTest, 'PCA', pca_df_train.shape[1])
CNN(pca_df_train, pca_df_test, yTrain, yTest, 'PCA', pca_df_train.shape[1])
RNN(pca_df_train, pca_df_test, yTrain, yTest, 'PCA', pca_df_train.shape[1])
