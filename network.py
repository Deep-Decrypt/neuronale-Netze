import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import json
import datetime # Import datetime

# Globale Variablen (könnten in Zukunft in Klassen umgewandelt werden, bleiben aber vorerst global für die Migration)
model = None
training_texts = []
data_array = None
labels_array = None
tokenizer = None
max_len = None

# Custom Callback zur Protokollierung der Learning Rate
class LearningRateLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        lr = self.model.optimizer.lr.numpy() # Aktuelle Learning Rate abrufen
        tf.summary.scalar('learning_rate', data=lr, step=epoch) # In TensorBoard protokollieren

# Funktion zum Erstellen eines Basismodells (kann in den spezifischen Skripten erweitert werden)
def create_base_model(vocab_size, max_len, embedding_dim=16, lstm_units=128, num_classes=26):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
        LSTM(lstm_units),
        Dense(num_classes, activation='softmax')
    ])
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    return model

# Modell und zugehörige Parameter speichern
def save_model_and_params(model, tokenizer, max_len, file_path):
    model.save(file_path)
    with open(file_path.replace('.h5', '_params.json'), 'w', encoding='utf-8') as f:
        json.dump({'max_len': max_len, 'tokenizer_config': tokenizer.to_json()}, f, indent=4)
    print(f"Modell und Parameter gespeichert in {file_path} und {file_path.replace('.h5', '_params.json')}")

# Modell und zugehörige Parameter laden
def load_model_and_params(file_path):
    # global model, tokenizer, max_len # No longer modify globals directly here
    if os.path.exists(file_path): # Only load if file exists and return None if not
        model = tf.keras.models.load_model(file_path)
        with open(file_path.replace('.h5', '_params.json'), 'r', encoding='utf-8') as f:
            params = json.load(f)
            tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(params['tokenizer_config'])
            max_len = params['max_len']
        print("Modell und Parameter geladen.")
        return model, tokenizer, max_len  # Return loaded values
    else:
        print("Modelldatei nicht gefunden. Bitte zuerst ein Modell erstellen oder trainieren.")
        return None, None, None # Return None values to indicate failure

# Funktion zum Laden der Trainingsdaten aus einer JSON-Datei
def load_training_data(file_path='training_data.json', default_texts=None):
    if not os.path.exists(file_path):
        if default_texts is None:
            default_texts = ["standard training text 1", "standard training text 2"] # Standardwerte, falls keine übergeben werden
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({"texts": default_texts}, f, indent=4, ensure_ascii=False)
        return default_texts

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data.get("texts", [])

# Speichern der Trainingsdaten und Aktualisierung der globalen Variablen
def update_training_data(new_texts, file_path='training_data.json', keep=False):
    global training_texts
    if keep:
        existing_texts = load_training_data(file_path)
        new_texts = list(set(existing_texts + new_texts))

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump({"texts": new_texts}, f, indent=4, ensure_ascii=False)
    training_texts = new_texts
    return training_texts # Rückgabe der aktualisierten Trainings-Texte

def prepare_data(texts, tokenizer=None, char_level=True):
    if tokenizer is None:
        tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=char_level)
        tokenizer.fit_on_texts(texts)
    data_seq = tokenizer.texts_to_sequences(texts)
    max_len = max(len(seq) for seq in data_seq) if data_seq else 0 # Handle case where data_seq is empty
    data_seq_padded = pad_sequences(data_seq, maxlen=max_len, padding='post')
    data_array = np.array(data_seq_padded)
    return data_array, tokenizer, max_len

def generate_tensorboard_callback(log_subdir):
    log_dir = os.path.join("logs", log_subdir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    return tensorboard_callback, log_dir

def print_tensorboard_instructions(log_dir):
    print(f"TensorBoard logs are saved in: {log_dir}")
    print("Um TensorBoard zu starten, öffne ein neues Terminal, navigiere zum Skriptverzeichnis und führe aus:")
    print(f"tensorboard --logdir logs/{os.path.basename(os.path.dirname(log_dir))}") # Dynamischer Pfad basierend auf log_dir