import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer # Explizit importieren, um Verwirrung zu vermeiden
import numpy as np
import string
import random
import os
import json
import datetime
from network import ( # Importiere Funktionen und Klassen aus network.py
    model, training_texts, data_array, labels_array, tokenizer, max_len,
    LearningRateLogger, create_base_model, save_model_and_params,
    load_model_and_params, load_training_data, update_training_data,
    prepare_data, generate_tensorboard_callback, print_tensorboard_instructions
)
from tensorflow.keras.preprocessing.sequence import pad_sequences # Import pad_sequences

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


# Alphabet für Caesar-Chiffre
alphabet = string.ascii_lowercase

# Funktion zur Generierung eines umfangreicheren Datensatzes
def generate_caesar_dataset(texts, shift_range=26):
    data = []
    labels = []
    for text in texts:
        for shift in range(1, shift_range):
            shifted_text = ''.join(
                alphabet[(alphabet.index(c) + shift) % len(alphabet)] if c in alphabet else c for c in text
            )
            data.append(shifted_text)
            labels.append(shift)
    return data, labels

# Modell erstellen (spezifisch für Caesar)
def create_caesar_model(tokenizer, max_len):
    return create_base_model(len(tokenizer.word_index)+1, max_len, embedding_dim=16, lstm_units=128, num_classes=26)


# Funktion zur Entschlüsselung beliebiger Texte
def decrypt_caesar(model, tokenizer, max_len, text):
    if model is None:
        print("Kein Modell geladen. Bitte zuerst ein Modell erstellen oder laden.")
        return None, None

    seq = tokenizer.texts_to_sequences([text])
    seq_padded = pad_sequences(seq, maxlen=max_len, padding='post')
    prediction = model.predict(seq_padded, verbose=0) # verbose=0 hinzugefügt
    predicted_shift = np.argmax(prediction[0])

    decrypted_text = ''.join(
        alphabet[(alphabet.index(c) - predicted_shift) % len(alphabet)] if c in alphabet else c for c in text
    )
    return decrypted_text, predicted_shift

def encrypt_caesar(text, shift):
    encrypted_text = ''.join(
        alphabet[(alphabet.index(c) + shift) % len(alphabet)] if c in alphabet else c for c in text
    )
    return encrypted_text

# Funktionen für die einzelnen Befehle
def command_erstellen():
    global model, tokenizer, max_len, data_array, labels_array
    default_texts = [ # Definiere Standardtexte lokal für Caesar
        "diesisteinegeheimebotschaft",
        "maschinelleslernenfetzt",
        "tensorflowmodelltraining",
        "neuronaleNetzesindstark",
        "kryptographieistinteressant"
    ]
    training_texts_local = load_training_data(default_texts=default_texts) # Verwende default_texts
    data, labels = generate_caesar_dataset(training_texts_local)

    data_array, tokenizer, max_len = prepare_data(data, char_level=True) # Nutze prepare_data Funktion

    model = create_caesar_model(tokenizer, max_len) # Nutze create_caesar_model
    print("Caesar Modell erstellt.")

def command_laden():
    global model, tokenizer, max_len, data_array, labels_array, training_texts
    loaded_model, loaded_tokenizer, loaded_max_len = load_model_and_params('caesar_model.h5') # Get returned values
    if loaded_model is not None: # Check if loading was successful
        model = loaded_model # Update global model
        tokenizer = loaded_tokenizer # Update global tokenizer
        max_len = loaded_max_len # Update global max_len

        training_texts = load_training_data()
        data, labels = generate_caesar_dataset(training_texts)
        data_array, tokenizer, max_len = prepare_data(data, tokenizer=tokenizer, char_level=True)
        labels_array = np.array(labels)
        print("Modell erfolgreich geladen und globale Variablen aktualisiert.") # Success message
    else:
        print("Modell konnte nicht geladen werden. Überprüfen Sie die Modelldatei.") # Error message


def command_trainieren():
    global model, data_array, labels_array, tokenizer, max_len
    if model is None or data_array is None or labels_array is None:
        print("Fehlendes Modell oder Trainingsdaten. Bitte zuerst ein Modell erstellen oder laden.")
        return

    epochs = int(input("Gib die Anzahl der Epochen ein: "))
    verbose_input = input("Soll die Ausgabe während des Trainings angezeigt werden? (J/n): ").lower()
    verbose = 1 if verbose_input == 'j' or verbose_input == 'y' else 0

    # Validation Data
    validation_texts = [
        "Das ist ein validierungssatz für das caesar chiffrier modell.",
        "Wir testen die genauigkeit des modells mit diesen saetzen.",
        "Funktioniert die entschlüsselung auch bei ganzen sätzen zuverlässig?",
        "Die validierungsdaten bestehen aus sinnvollen deutschen sätzen.",
        "Dieser satz dient dazu die generalisierung des modells zu überprüfen."
    ] # Validierungssätze
    validation_data_raw, validation_labels_raw = generate_caesar_dataset(validation_texts)
    validation_data_array, _, _ = prepare_data(validation_data_raw, tokenizer=tokenizer, char_level=True) # Nutze prepare_data
    validation_labels_array = np.array(validation_labels_raw)

    # TensorBoard Callback
    tensorboard_callback, log_dir = generate_tensorboard_callback("caesar_fit") # Nutze generate_tensorboard_callback

    # Learning Rate Logger Callback
    lr_logger_callback = LearningRateLogger() # Callback instanziieren

    model.fit(data_array, labels_array, epochs=epochs, batch_size=500, verbose=verbose,
              validation_data=(validation_data_array, validation_labels_array), # validation_data hinzugefügt
              callbacks=[tensorboard_callback, lr_logger_callback]) # Beide Callbacks hinzufügen
    print("Modell trainiert.")
    print_tensorboard_instructions(log_dir) # Nutze print_tensorboard_instructions


def command_entschluesseln():
    global model, tokenizer, max_len
    if model is None:
        print("Kein Modell geladen. Modell ist None.") # Debug print
        print("Bitte zuerst ein Modell erstellen oder laden.")
        return
    cipher_text = input("Gib den verschlüsselten Text ein: ").lower()
    decrypted_text, key = decrypt_caesar(model, tokenizer, max_len, cipher_text)
    if decrypted_text:
        print(f"Entschlüsselter Text: {decrypted_text}")
        print(f"Verwendeter Schlüssel (Shift-Wert): {key}")

def entschluesseln(text):
    global model, tokenizer, max_len
    if model is None:
        print("Kein Modell geladen. Bitte zuerst ein Modell erstellen oder laden.")
        return None, None
    cipher_text = text
    decrypted_text, key = decrypt_caesar(model, tokenizer, max_len, cipher_text)

    return decrypted_text, key

def command_texte_bearbeiten():
    global tokenizer, max_len, data_array, labels_array
    new_texts_input = input("Gib die neuen Trainings-Texte, kommasepariert, ein: ").lower()
    new_texts = [text.strip() for text in new_texts_input.split(',') if len(text.strip()) >= 2] # Kein lower() hier, um Großbuchstaben zu behalten, falls relevant
    if not new_texts:
        print("Keine neuen Texte eingegeben.")
        return

    updated_texts = update_training_data(new_texts, keep=True) # Nutze update_training_data
    data, labels = generate_caesar_dataset(updated_texts) # Datensatz neu generieren
    data_array, tokenizer, max_len = prepare_data(data, tokenizer=tokenizer, char_level=True) # Bereite Daten mit (möglicherweise) neuem Tokenizer vor
    labels_array = np.array(labels)
    print("Trainings-Texte aktualisiert.")

def command_sichern():
    global model, tokenizer, max_len
    if model is None:
        print("Kein Modell geladen. Bitte zuerst ein Modell erstellen oder laden.")
        return
    save_model_and_params(model, tokenizer, max_len, 'caesar_model.h5') # Nutze save_model_and_params

def command_hilfe():
    print("Mögliche Befehle: 'erstellen', 'laden', 'trainieren', 'entschlüsseln', 'texte_bearbeiten', 'sichern', 'beenden'")

def command_beenden():
    global model, tokenizer, max_len
    if model is not None:
        save = input("Möchtest du das Modell und die Parameter speichern? (j/N): ").lower()
        if save == 'j' or save == 'y':
            save_model_and_params(model, tokenizer, max_len, 'caesar_model.h5') # Nutze save_model_and_params
    print("Programm wird beendet.")
    return True  # Signal zum Beenden der Hauptschleife

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
    }

    while True:
        command = input("Gib einen Befehl ein: ").lower()
        if command in command_map:
            if command_map[command]():  # Ausführen der Funktion und prüfen auf Beenden-Signal
                break
        else:
            print("Ungültiger Befehl. Bitte gib 'hilfe' für eine Übersicht der Befehle ein.")