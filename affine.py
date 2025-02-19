import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer # Explizit importieren, um Verwirrung zu vermeiden
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # Importiere spezifische Callbacks hier
import numpy as np
import string
import random
import os
import json
import re
import datetime
from network import ( # Importiere Funktionen und Klassen aus network.py
    model, training_texts, data_array, labels_array, tokenizer, max_len,
    LearningRateLogger, create_base_model, save_model_and_params,
    load_model_and_params, load_training_data, update_training_data,
    prepare_data, generate_tensorboard_callback, print_tensorboard_instructions
)
from tensorflow.keras.optimizers import Adam # Importiere Adam hier, falls nicht bereits importiert
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding # Importiere benötigte Layer
from tensorflow.keras.preprocessing.sequence import pad_sequences # Importiere pad_sequences


VALID_A = [1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25]
NUM_ENCRYPTIONS_PER_TEXT = 300  # Reduziert, um Überlappung zu vermeiden
NUM_TEST_SAMPLES = 5  # Anzahl der Testfälle

# Hilfsfunktion für Umlaut-Ersetzung
def replace_umlauts(text):
    replacements = {
        'ä': 'ae',
        'ö': 'oe',
        'ü': 'ue',
        'ß': 'ss',
        'Ä': 'Ae',
        'Ö': 'Oe',
        'Ü': 'Ue',
        'ß': 'ss'
    }
    return "".join(replacements.get(char, char) for char in text)


# Hilfsfunktionen für Affine Cipher
def modular_inverse(a, m):
    for x in range(1, m):
        if (a * x) % m == 1:
            return x
    return None

def affine_encrypt(text, a, b):
    encrypted = []
    for char in text:
        if char.isalpha():
            is_upper = char.isupper()
            x = ord(char.upper()) - ord('A')
            encrypted_x = (a * x + b) % 26
            encrypted_char = chr(encrypted_x + ord('A'))
            if not is_upper:
                encrypted_char = encrypted_char.lower()
            encrypted.append(encrypted_char)
        else:
            encrypted.append(char)
    return "".join(encrypted)

def affine_decrypt(ciphertext, a, b):
    decrypted = []
    a_inv = modular_inverse(a, 26)
    for char in ciphertext:
        if char.isalpha():
            is_upper = char.isupper()
            y = ord(char.upper()) - ord('A')
            decrypted_y = (a_inv * (y - b)) % 26
            decrypted_char = chr(decrypted_y + ord('A'))
            if not is_upper:
                decrypted_char = decrypted_char.lower()
            decrypted.append(decrypted_char)
        else:
            decrypted.append(char)
    return "".join(decrypted)

# Datensatzgenerierung für Affine Cipher
def generate_affine_dataset(texts):
    data = []
    labels = []
    for text in texts:
        for a in VALID_A:
            for b in range(26):
                encrypted = affine_encrypt(text, a, b)
                a_idx = VALID_A.index(a)
                label = a_idx * 26 + b
                if encrypted.lower() not in data:
                    data.append(encrypted.lower())
                    labels.append(label)
    return data, labels

# Modellarchitektur
def create_affine_model(tokenizer, max_len):
    model = Sequential([
        Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=64, input_length=max_len),
        LSTM(256, return_sequences=True), # return_sequences=True für Stacked LSTM
        Dropout(0.2),
        LSTM(256), # Stacked LSTM Layer
        Dropout(0.2),
        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(312, activation='softmax')
    ])
    optimizer = Adam(learning_rate=0.001)  # Reduzierte Learning Rate
    model.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    return model

# Entschlüsselungsfunktion
def decrypt_affine(model, tokenizer, max_len, ciphertext):
    ciphertext = ciphertext.lower()
    seq = tokenizer.texts_to_sequences([ciphertext])
    seq_padded = pad_sequences(seq, maxlen=max_len, padding='post')
    prediction = model.predict(seq_padded, verbose=0) # verbose=0 hinzugefügt
    label = np.argmax(prediction[0])

    a_idx = label // 26
    b = label % 26
    a = VALID_A[a_idx]

    return affine_decrypt(ciphertext, a, b), (a, b)

# Angepasste Befehle
def command_erstellen():
    global model, tokenizer, max_len, data_array, labels_array
    default_texts = [ # Definiere Standardtexte lokal für Affine
        "dies ist eine geheime botschaft",
        "maschinelles lernen fetzt",
        "tensorflow modell training",
        "neuronale netze sind stark",
        "kryptographie ist interessant",
        "persoenlich sind unsere entscheidungen"
    ]
    training_texts_local = load_training_data(default_texts=default_texts) # Nutze load_training_data aus network.py und verwende default_texts
    training_texts_local = [replace_umlauts(text) for text in training_texts_local] # Umlautersetzung
    data, labels = generate_affine_dataset(training_texts_local)

    data_array, tokenizer, max_len = prepare_data(data, char_level=True) # Nutze prepare_data Funktion

    model = create_affine_model(tokenizer, max_len) # Nutze create_affine_model
    print("Affine Cipher Modell erstellt.")

    # Testausgabe nach dem Erstellen
    print("\nTestausgabe:")
    _test_model()


def command_trainieren():
    global model, data_array, labels_array, tokenizer, max_len
    if model is None:
        print("Bitte zuerst Modell erstellen oder laden!")
        return

    epochs = int(input("Epochen: "))
    batch_size = int(input("Batch Size: "))
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True) # Monitor validation loss
    model_checkpoint = ModelCheckpoint('best_affine_model.h5', monitor='val_loss', save_best_only=True) # Monitor validation loss

    # Validierungsdaten
    validation_texts = [
        "Das ist ein validierungssatz für das affine chiffrier modell.",
        "Wir testen die genauigkeit des modells mit diesen saetzen.",
        "Funktioniert die entschlüsselung auch bei ganzen sätzen zuverlässig?",
        "Die validierungsdaten bestehen aus sinnvollen deutschen sätzen.",
        "Dieser satz dient dazu die generalisierung des modells zu überprüfen."
    ] # Validierungssätze
    validation_texts = [replace_umlauts(text) for text in validation_texts] # Umlautersetzung
    validation_data_raw, validation_labels_raw = generate_affine_dataset(validation_texts)
    validation_data_array, _, _ = prepare_data(validation_data_raw, tokenizer=tokenizer, char_level=True) # Nutze prepare_data
    validation_labels_array = np.array(validation_labels_raw)

    # TensorBoard Callback
    tensorboard_callback, log_dir = generate_tensorboard_callback("affine_fit") # Nutze generate_tensorboard_callback

    # Learning Rate Logger Callback
    lr_logger_callback = LearningRateLogger() # Callback instanziieren

    model.fit(data_array, labels_array, epochs=epochs, batch_size=batch_size, verbose=1,
              validation_data=(validation_data_array, validation_labels_array), # validation_data hinzugefügt
              callbacks=[early_stopping, model_checkpoint, tensorboard_callback, lr_logger_callback]) # Alle Callbacks hinzufügen
    print("Modell trainiert.")
    print_tensorboard_instructions(log_dir) # Nutze print_tensorboard_instructions


def command_entschluesseln():
    global model, tokenizer, max_len
    if model is None:
        print("Kein Modell geladen. Bitte zuerst ein Modell erstellen oder laden.")
        return
    cipher_text = input("Gib den verschlüsselten Text ein: ")
    cipher_text = replace_umlauts(cipher_text).lower() # Umlautersetzung und Leerzeichen entfernen
    decrypted_text, (a, b) = decrypt_affine(model, tokenizer, max_len, cipher_text)
    if decrypted_text:
        print(f"Entschlüsselter Text: {decrypted_text}")
        print(f"Verwendeter Schlüssel (a={a}, b={b})")

def entschluesseln(text):
    global model, tokenizer, max_len
    if model is None:
        print("Kein Modell geladen. Bitte zuerst ein Modell erstellen oder laden.")
        return
    cipher_text = text
    decrypted_text, key = decrypt_affine(model, tokenizer, max_len, cipher_text)
    if decrypted_text:
        return decrypted_text
    return None

def command_texte_bearbeiten():
    global tokenizer, max_len, data_array, labels_array
    new_texts_input = input("Gib die neuen Trainings-Texte, kommasepariert, ein: ")
    new_texts = [text.strip().lower() for text in new_texts_input.split(',') if len(text.strip()) >= 2]
    if not new_texts:
        print("Keine neuen Texte eingegeben.")
        return

    new_texts = [replace_umlauts(text) for text in new_texts] # Umlautersetzung

    updated_texts = update_training_data(new_texts, keep=True) # Nutze update_training_data
    data, labels = generate_affine_dataset(updated_texts) # Datensatz neu generieren
    data_array, tokenizer, max_len = prepare_data(data, tokenizer=tokenizer, char_level=True) # Bereite Daten mit (möglicherweise) neuem Tokenizer vor
    labels_array = np.array(labels)
    print("Trainings-Texte aktualisiert.")

def command_sichern():
    global model, tokenizer, max_len
    if model is None:
        print("Kein Modell geladen. Bitte zuerst ein Modell erstellen oder laden.")
        return
    save_model_and_params(model, tokenizer, max_len, 'affine_model.h5') # Nutze save_model_and_params

def command_hilfe():
    print("Verfügbare Befehle: 'erstellen' oder 'neu', 'laden', 'trainieren', 'entschlüsseln', 'texte_bearbeiten', 'sichern', 'beenden' oder 'exit', 'neu_verschluesseln' oder 'help'")

def command_beenden():
    global model, tokenizer, max_len
    if model is not None:
        save = input("Möchtest du das Modell und die Parameter speichern? (j/N): ").lower()
        if save == 'j' or save == 'y':
            save_model_and_params(model, tokenizer, max_len, 'affine_model.h5') # Nutze save_model_and_params
    print("Programm wird beendet.")
    return True  # Signal zum Beenden der Hauptschleife

def command_laden():
    global model, tokenizer, max_len, data_array, labels_array, training_texts
    loaded_model, loaded_tokenizer, loaded_max_len = load_model_and_params('affine_model.h5') # Rückgabewerte empfangen
    if loaded_model is not None: # Nutze load_model_and_params aus network.py und prüfe auf Erfolg
        model = loaded_model # Globale model Variable aktualisieren
        tokenizer = loaded_tokenizer # Globale tokenizer Variable aktualisieren
        max_len = loaded_max_len # Globale max_len Variable aktualisieren

        training_texts = load_training_data() # Lade Trainingsdaten nach dem Modelladen
        training_texts = [replace_umlauts(text) for text in training_texts] # Umlautersetzung
        data, labels = generate_affine_dataset(training_texts) # Generiere Datensatz neu
        data_array, tokenizer, max_len = prepare_data(data, tokenizer=tokenizer, char_level=True) # Bereite Daten mit geladenem Tokenizer vor
        labels_array = np.array(labels) # Labels Array setzen
        # Testausgabe nach dem Laden
        print("\nTestausgabe:")
        _test_model()
    else:
        print("Laden des Modells fehlgeschlagen.")


# Hilfsfunktion zum Testen des Modells
def _test_model():
    global data_array, labels_array, tokenizer, max_len, model
    if data_array is None or labels_array is None or tokenizer is None or max_len is None or model is None:
        print("Keine Trainingsdaten oder Modell vorhanden.")
        return

    num_samples = min(NUM_TEST_SAMPLES, len(data_array))
    indices = random.sample(range(len(data_array)), num_samples)

    for idx in indices:
        cipher_text = "".join([tokenizer.index_word.get(i, '') for i in data_array[idx] if i != 0])
        decrypted_text, (a, b) = decrypt_affine(model, tokenizer, max_len, cipher_text)
        print(f"Verschlüsselt: {cipher_text}")
        print(f"Entschlüsselt: {decrypted_text}")
        print(f"Schlüssel (a={a}, b={b})\n")

def command_neu_verschluesseln():
    global training_texts, data_array, labels_array, tokenizer, max_len
    if not training_texts:
        print("Keine Trainings-Texte vorhanden. Bitte zuerst Texte laden oder erstellen.")
        return

    training_texts = [replace_umlauts(text) for text in training_texts] # Umlautersetzung
    data, labels = generate_affine_dataset(training_texts)

    data_array, tokenizer, max_len = prepare_data(data, tokenizer=tokenizer, char_level=True) # Nutze prepare_data
    labels_array = np.array(labels)
    print("Texte wurden neu verschlüsselt und der Datensatz wurde aktualisiert.")

# Hauptschleife des Programms
if __name__ == "__main__":
    command_map = {
        'erstellen': command_erstellen,
        'neu': command_erstellen,
        'new': command_erstellen,
        'create': command_erstellen,
        'laden': command_laden,
        'load': command_laden,
        'trainieren': command_trainieren,
        'train': command_trainieren,
        'fit': command_trainieren,
        'entschlüsseln': command_entschluesseln,
        'decrypt': command_entschluesseln,
        'texte_bearbeiten': command_texte_bearbeiten,
        'edit': command_texte_bearbeiten,
        'sichern': command_sichern,
        'save': command_sichern,
        'hilfe': command_hilfe,
        'help': command_hilfe,
        '?': command_hilfe,
        'beenden': command_beenden,
        'exit': command_beenden,
        'neu_verschluesseln': command_neu_verschluesseln,
    }

    while True:
        try:
          command = input("Gib einen Befehl ein: ").lower()
          if command in command_map:
              if command_map[command]():  # Ausführen der Funktion und prüfen auf Beenden-Signal
                  break
          else:
              print("Ungültiger Befehl. Bitte gib 'hilfe' oder '?' für eine Übersicht der Befehle ein.")
        except Exception as e:
          print(f"Ein Fehler ist aufgetreten: {e}")