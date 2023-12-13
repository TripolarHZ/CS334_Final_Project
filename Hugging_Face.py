from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('data.csv')
df = data.drop('datetime', axis = 1, inplace=True)
desired_label = ['depression', 'neutral']
df = data[data['label'].isin(desired_label)]
df = df.dropna(subset=['content'])
df = df[df['content'].apply(lambda x: isinstance(x, str))]

def encode_label(label):
    if label == 'depression':
        return 1
    else:
        return 0

df['encoded_label'] = df['label'].apply(encode_label)

x = pd.DataFrame(df['content'])
y = pd.DataFrame(df['encoded_label'])
xTrain,xTest,yTrain,yTest = train_test_split(x, y, train_size=0.7, shuffle=True)
xTrain = xTrain.head(10000)
xTest = xTest.head(10000)
yTrain = yTrain.head(10000)
yTest = yTest.head(10000)
print('Finished Splitting...')

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples, padding="max_length", truncation=True, max_length = 64)

xTrain_tokenized = tokenize_function(xTrain["content"].tolist())
xTest_tokenized = tokenize_function(xTest["content"].tolist())
print('Finished Tokenizing...')

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(xTrain_tokenized),
    yTrain
))
test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(xTest_tokenized),
    yTest
))
print('Finished Converting...')

model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
print('Finished Loading...')

num_train_steps = len(train_dataset) * 3
warmup_steps = int(0.1 * num_train_steps)

# Create a learning rate scheduler
learning_rate_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=5e-5,
    decay_steps=num_train_steps,
    end_learning_rate=0.0
)

# Create the optimizer
optimizer = tf.keras.optimizers.Adam(
    learning_rate=learning_rate_scheduler
)

model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=tf.metrics.SparseCategoricalAccuracy())

model.fit(train_dataset.shuffle(len(xTrain)).batch(32).prefetch(tf.data.AUTOTUNE), epochs=5)
print('Finished Training...')

loss, accuracy = model.evaluate(test_dataset.batch(32))
print(f"Test accuracy: {accuracy*100}%")