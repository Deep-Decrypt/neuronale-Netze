import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, scrolledtext
import caesar
import affine
import threading
import os
import ttkbootstrap as ttkb

current_theme = "darkly"

class CipherUI:
    def __init__(self, master):
        self.master = master
        master.title("Cipher Tool") # Genereller Titel
        master.geometry("1100x750")

        self.caesar_module = caesar
        self.affine_module = affine # Importiere und speichere das affine Modul

        self.current_module = self.caesar_module # Standardmäßig Caesar Modul
        self.cipher_type = "Caesar" # Starte mit Caesar als Standard

        self.training_texts = []
        self.data_array = None
        self.labels_array = None
        self.tokenizer = None
        self.max_len = None
        self.model = None
        self.preview_text = None
        self.encrypted_text_display = None # For encryption tab output
        self.encryption_key_display_label = None # Genereller Label für Schlüsselanzeige

        self.style = ttkb.Style(theme=current_theme)
        self.style.configure('.', font=('Helvetica', 11))
        self.style.configure('TNotebook.Tab', font=('Helvetica', 12))
        self.style.configure('TButton', padding=10)

        padding_x = 15
        padding_y = 10
        group_padding_y = 20

        self.notebook = ttkb.Notebook(master, padding=(padding_x, padding_y))
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=padding_x, pady=padding_y)

        self.decryption_tab = ttkb.Frame(self.notebook, padding=(padding_x, padding_y))
        self.notebook.add(self.decryption_tab, text='Entschlüsselung')

        self.encryption_tab = ttkb.Frame(self.notebook, padding=(padding_x, padding_y))
        self.notebook.add(self.encryption_tab, text='Verschlüsselung')

        self.model_tab = ttkb.Frame(self.notebook, padding=(padding_x, padding_y))
        self.notebook.add(self.model_tab, text='Modellverwaltung')

        self.create_decryption_tab_content(self.decryption_tab, padding_x, padding_y, group_padding_y)
        self.create_encryption_tab_content(self.encryption_tab, padding_x, padding_y, group_padding_y)
        self.create_model_tab_content(self.model_tab, padding_x, padding_y, group_padding_y)

        self.status_label = ttkb.Label(master, text="Bereit", anchor=tk.W, padding=(padding_x, padding_y))
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        self.create_menu()

        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

        self.load_model() # Lade Modell beim Start

    def on_tab_changed(self, event):
        current_tab = self.notebook.nametowidget(self.notebook.select())
        current_tab.focus_set()

    def create_menu(self):
        menubar = tk.Menu(self.master)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Beenden", command=self.master.quit)
        menubar.add_cascade(label="Datei", menu=filemenu)

        thememenu = tk.Menu(menubar, tearoff=0)
        themes = ["litera", "flatly", "darkly", "solarized", "minty", "pulse", "morph", "cerculean", "journal", "united", "yeti", "sandstone", "superhero"]
        for theme in themes:
            thememenu.add_command(label=theme, command=lambda t=theme: self.set_theme(t))
        menubar.add_cascade(label="Design", menu=thememenu)

        self.master.config(menu=menubar)

    def set_theme(self, theme):
        global current_theme
        current_theme = theme
        self.style.theme_use(theme)
        print(f"Theme set to {theme}")

    def create_decryption_tab_content(self, tab, padding_x, padding_y, group_padding_y):
        tab.columnconfigure(0, weight=1)

        input_frame = ttkb.Frame(tab, padding=(padding_x, group_padding_y))
        input_frame.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, padding_x//2))
        input_frame.columnconfigure(0, weight=1)
        input_frame.rowconfigure(1, weight=1)

        input_label = ttkb.Label(input_frame, text="Eingabetext:")
        input_label.grid(row=0, column=0, sticky=tk.W, pady=(0, padding_y//2))
        self.input_text = scrolledtext.ScrolledText(input_frame, height=10, width=60, wrap=tk.WORD)
        self.input_text.grid(row=1, column=0, sticky=tk.NSEW)
        self.input_text.bind("<Control-a>", self.select_all)
        self.input_text.bind("<Control-c>", lambda event: self.master.focus_get().event_generate("<<Copy>>"))
        self.input_text.bind("<Control-v>", self.paste_text)
        self.input_text.bind("<Control-x>", lambda event: self.master.focus_get().event_generate("<<Cut>>"))
        self.input_text.bind("<KeyRelease>", self.update_preview)

        for child in self.input_text.winfo_children():
            if isinstance(child, tk.Scrollbar):
                if child.cget('orient') == 'vertical':
                    child.config(width=0)
                elif child.cget('orient') == 'horizontal':
                    child.config(height=0)

        preview_frame = ttkb.Frame(tab, padding=(padding_x, group_padding_y))
        preview_frame.grid(row=1, column=0, columnspan=2, sticky=tk.NSEW)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(1, weight=1)

        preview_label = ttkb.Label(preview_frame, text="Vorschau Entschlüsselung:")
        preview_label.grid(row=0, column=0, sticky=tk.W, pady=(0, padding_y//2))
        self.preview_text = scrolledtext.ScrolledText(preview_frame, height=10, width=120, wrap=tk.WORD, state=tk.DISABLED)
        self.preview_text.grid(row=1, column=0, sticky=tk.NSEW)

        for child in self.preview_text.winfo_children():
            if isinstance(child, tk.Scrollbar):
                if child.cget('orient') == 'vertical':
                    child.config(width=0)
                elif child.cget('orient') == 'horizontal':
                    child.config(height=0)

        button_frame = ttkb.Frame(tab, padding=(padding_x, group_padding_y))
        button_frame.grid(row=2, column=0, columnspan=2, sticky=tk.EW)
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

        decrypt_button = ttkb.Button(button_frame, text="Entschlüsseln", command=self.decrypt, bootstyle="success")
        decrypt_button.grid(row=0, column=0, padx=(0, padding_x//2), pady=padding_y, sticky=tk.E+tk.W)

        clear_button = ttkb.Button(button_frame, text="Leeren", command=self.clear_decryption, bootstyle="secondary")
        clear_button.grid(row=0, column=1, padx=(padding_x//2, 0), pady=padding_y, sticky=tk.W+tk.E)

        self.key_display_label = ttkb.Label(tab, text="", font=('Helvetica', 11, 'italic')) # Genereller Key Display Label
        self.key_display_label.grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=padding_x, pady=(0, group_padding_y))

    def create_encryption_tab_content(self, tab, padding_x, padding_y, group_padding_y):
        tab.columnconfigure(0, weight=1)

        input_frame = ttkb.Frame(tab, padding=(padding_x, group_padding_y))
        input_frame.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, padding_x//2))
        input_frame.columnconfigure(0, weight=1)
        input_frame.rowconfigure(1, weight=1)

        input_label = ttkb.Label(input_frame, text="Klartext:")
        input_label.grid(row=0, column=0, sticky=tk.W, pady=(0, padding_y//2))
        self.encryption_input_text = scrolledtext.ScrolledText(input_frame, height=10, width=60, wrap=tk.WORD)
        self.encryption_input_text.grid(row=1, column=0, sticky=tk.NSEW)
        self.encryption_input_text.bind("<Control-a>", self.select_all)
        self.encryption_input_text.bind("<Control-c>", lambda event: self.master.focus_get().event_generate("<<Copy>>"))
        self.encryption_input_text.bind("<Control-v>", self.paste_text)
        self.encryption_input_text.bind("<Control-x>", lambda event: self.master.focus_get().event_generate("<<Cut>>"))

        for child in self.encryption_input_text.winfo_children():
            if isinstance(child, tk.Scrollbar):
                if child.cget('orient') == 'vertical':
                    child.config(width=0)
                elif child.cget('orient') == 'horizontal':
                    child.config(height=0)

        output_frame = ttkb.Frame(tab, padding=(padding_x, group_padding_y))
        output_frame.grid(row=1, column=0, columnspan=2, sticky=tk.NSEW)
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(1, weight=1)

        output_label = ttkb.Label(output_frame, text="Verschlüsselter Text:")
        output_label.grid(row=0, column=0, sticky=tk.W, pady=(0, padding_y//2))
        self.encrypted_text_display = scrolledtext.ScrolledText(output_frame, height=10, width=120, wrap=tk.WORD, state=tk.DISABLED)
        self.encrypted_text_display.grid(row=1, column=0, sticky=tk.NSEW)

        for child in self.encrypted_text_display.winfo_children():
            if isinstance(child, tk.Scrollbar):
                if child.cget('orient') == 'vertical':
                    child.config(width=0)
                elif child.cget('orient') == 'horizontal':
                    child.config(height=0)

        button_frame = ttkb.Frame(tab, padding=(padding_x, group_padding_y))
        button_frame.grid(row=2, column=0, columnspan=2, sticky=tk.EW)
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(2, weight=1) # For copy button

        encrypt_button = ttkb.Button(button_frame, text="Verschlüsseln", command=self.encrypt_text, bootstyle="success")
        encrypt_button.grid(row=0, column=0, padx=(0, padding_x//2), pady=padding_y, sticky=tk.E+tk.W)

        clear_button = ttkb.Button(button_frame, text="Leeren", command=self.clear_encryption, bootstyle="secondary")
        clear_button.grid(row=0, column=1, padx=(padding_x//2, padding_x//2), pady=padding_y, sticky=tk.W+tk.E)

        copy_button = ttkb.Button(button_frame, text="Kopieren", command=self.copy_encrypted_text, bootstyle="info")
        copy_button.grid(row=0, column=2, padx=(padding_x//2, 0), pady=padding_y, sticky=tk.W+tk.E)

        self.encryption_key_display_label = ttkb.Label(tab, text="", font=('Helvetica', 11, 'italic')) # Genereller Key Display Label
        self.encryption_key_display_label.grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=padding_x, pady=(0, group_padding_y))

        # Frame für Chiffre-spezifische Eingaben (zunächst leer)
        self.cipher_input_frame = ttkb.Frame(tab, padding=(padding_x, padding_y))
        self.cipher_input_frame.grid(row=3, column=0, columnspan=2, sticky=tk.EW, padx=padding_x)

    def create_model_tab_content(self, tab, padding_x, padding_y, group_padding_y):
        buttons_frame = ttkb.Frame(tab, padding=(padding_x, group_padding_y))
        buttons_frame.pack(fill=tk.BOTH, expand=True)
        buttons_frame.columnconfigure(0, weight=1)
        buttons_frame.columnconfigure(1, weight=1)
        buttons_frame.columnconfigure(2, weight=1) # New column for Cipher Type

        create_button = ttkb.Button(buttons_frame, text="Modell erstellen", command=self.create_model, bootstyle="success")
        create_button.grid(row=0, column=0, padx=padding_x//2, pady=padding_y, sticky=tk.W+tk.E)

        load_button = ttkb.Button(buttons_frame, text="Modell laden", command=self.load_model, bootstyle="primary")
        load_button.grid(row=0, column=1, padx=padding_x//2, pady=padding_y, sticky=tk.W+tk.E)

        train_button = ttkb.Button(buttons_frame, text="Modell trainieren", command=self.train_model, bootstyle="info")
        train_button.grid(row=1, column=0, padx=padding_x//2, pady=padding_y, sticky=tk.W+tk.E)

        save_button = ttkb.Button(buttons_frame, text="Modell speichern", command=self.save_model, bootstyle="warning")
        save_button.grid(row=1, column=1, padx=padding_x//2, pady=padding_y, sticky=tk.W+tk.E)

        # Cipher Type Combobox
        cipher_label = ttkb.Label(buttons_frame, text="Chiffre Typ:")
        cipher_label.grid(row=0, column=2, padx=(group_padding_y, padding_x//2), pady=padding_y, sticky=tk.W)
        self.cipher_type_var = tk.StringVar(value=self.cipher_type) # Use self.cipher_type for initial value
        cipher_type_combobox = ttkb.Combobox(buttons_frame, textvariable=self.cipher_type_var, values=["Caesar", "Affine"], state="readonly")
        cipher_type_combobox.grid(row=1, column=2, padx=(group_padding_y, padding_x//2), pady=padding_y, sticky=tk.W+tk.E)
        cipher_type_combobox.bind("<<ComboboxSelected>>", self.on_cipher_type_changed)


    def on_cipher_type_changed(self, event):
        selected_cipher = self.cipher_type_var.get()
        if selected_cipher == "Caesar":
            self.current_module = self.caesar_module
            self.cipher_type = "Caesar"
        elif selected_cipher == "Affine":
            self.current_module = self.affine_module
            self.cipher_type = "Affine"
        print(f"Chiffre-Typ geändert zu: {self.cipher_type}")
        self.update_encryption_input_ui() # UI für Verschlüsselungstab aktualisieren

    def update_encryption_input_ui(self):
        # Leere zuerst den cipher_input_frame
        for widget in self.cipher_input_frame.winfo_children():
            widget.destroy()

        if self.cipher_type == "Caesar":
            shift_label = ttkb.Label(self.cipher_input_frame, text="Shift-Wert eingeben (0-25):")
            shift_label.pack(side=tk.LEFT, padx=5)
            self.shift_entry = ttkb.Entry(self.cipher_input_frame, width=5)
            self.shift_entry.insert(0, "3") # Standardwert für Caesar Shift
            self.shift_entry.pack(side=tk.LEFT)
        elif self.cipher_type == "Affine":
            a_label = ttkb.Label(self.cipher_input_frame, text="Wert für 'a' eingeben:")
            a_label.pack(side=tk.LEFT, padx=5)
            self.a_entry = ttkb.Entry(self.cipher_input_frame, width=5)
            self.a_entry.insert(0, "5") # Standardwert für Affine 'a'
            self.a_entry.pack(side=tk.LEFT)

            b_label = ttkb.Label(self.cipher_input_frame, text="Wert für 'b' eingeben (0-25):")
            b_label.pack(side=tk.LEFT, padx=5)
            self.b_entry = ttkb.Entry(self.cipher_input_frame, width=5)
            self.b_entry.insert(0, "8") # Standardwert für Affine 'b'
            self.b_entry.pack(side=tk.LEFT)


    def select_all(self, event):
        event.widget.tag_add("sel", "1.0", "end")
        event.widget.mark_set(tk.INSERT, "1.0")
        event.widget.see(tk.INSERT)
        return 'break'

    def set_status(self, message):
        self.status_label.config(text=message)
        self.master.update_idletasks()

    def create_model(self):
        try:
            self.set_status("Modell wird erstellt...")
            if self.cipher_type == "Caesar":
                self.current_module.command_erstellen() # Verwende current_module
            elif self.cipher_type == "Affine":
                self.current_module.command_erstellen() # Verwende current_module
            self.load_from_module()
            self.set_status("Modell erstellt.")
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Erstellen des Modells: {e}")
            self.set_status("Fehler")

    def load_model(self):
        model_file_path = f'{self.cipher_type.lower()}_model.h5' # Dynamischer Dateiname
        if not os.path.exists(model_file_path):
            messagebox.showerror("Fehler", f"Modelldatei nicht gefunden: {model_file_path}. Bitte stellen Sie sicher, dass die Datei im selben Verzeichnis liegt.")
            self.set_status("Modelldatei nicht gefunden.")
            return

        try:
            self.set_status("Modell wird geladen...")
            loaded_model, loaded_tokenizer, loaded_max_len = self.current_module.load_model_and_params(model_file_path) # Verwende current_module
            if loaded_model is not None:
                self.current_module.model = loaded_model # Verwende current_module
                self.current_module.tokenizer = loaded_tokenizer # Verwende current_module
                self.current_module.max_len = loaded_max_len # Verwende current_module
                self.load_from_module()
                self.set_status("Modell geladen.")
            else:
                messagebox.showerror("Fehler", "Fehler beim Laden des Modells (Details in der Konsole).")
                self.set_status("Fehler beim Laden des Modells.")
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Laden des Modells: {e}")
            self.set_status("Fehler")


    def train_model(self):
        if self.current_module.model is None: # Verwende current_module
            messagebox.showerror("Fehler", "Kein Modell geladen. Erstellen oder laden Sie zuerst ein Modell.")
            return

        def train_thread():
            try:
                self.set_status("Modell wird trainiert...")
                epochs_str = simpledialog.askstring("Training", "Anzahl der Epochen eingeben:", initialvalue="10")
                if epochs_str is None:
                    self.set_status("Training abgebrochen.")
                    return

                try:
                    epochs = int(epochs_str)
                except ValueError:
                    messagebox.showerror("Fehler", "Ungültige Anzahl von Epochen.")
                    self.set_status("Training abgebrochen.")
                    return

                batch_size_str = simpledialog.askstring("Training", "Batch Size eingeben:", initialvalue="32")
                if batch_size_str is None:
                    self.set_status("Training abgebrochen.")
                    return

                try:
                    batch_size = int(batch_size_str)
                except ValueError:
                    messagebox.showerror("Fehler", "Ungültige Batch Size.")
                    self.set_status("Training abgebrochen.")
                    return

                verbose_bool = messagebox.askyesno("Training", "Trainingsausgabe anzeigen?")
                verbose = 1 if verbose_bool else 0

                self.current_module.model.fit(self.current_module.data_array, self.current_module.labels_array, epochs=epochs, batch_size=batch_size, verbose=verbose) # Verwende current_module
                self.set_status("Modell trainiert.")
            except Exception as e:
                messagebox.showerror("Fehler", f"Fehler beim Trainieren des Modells: {e}")
                self.set_status("Fehler")

        threading.Thread(target=train_thread).start()

    def update_preview(self, event=None):
        if self.current_module.model is None: # Verwende current_module
            self.preview_text.config(state=tk.NORMAL)
            self.preview_text.delete("1.0", tk.END)
            self.preview_text.insert("1.0", "Modell nicht geladen")
            self.preview_text.config(state=tk.DISABLED)
            return

        input_text = self.input_text.get("1.0", tk.END).strip().lower()
        if not input_text:
            self.preview_text.config(state=tk.NORMAL)
            self.preview_text.delete("1.0", tk.END)
            self.preview_text.config(state=tk.DISABLED)
            return

        try:
            if self.cipher_type == "Caesar":
                preview_decrypted_text, _ = self.current_module.decrypt_caesar(self.current_module.model, self.current_module.tokenizer, self.current_module.max_len, input_text) # Verwende current_module
            elif self.cipher_type == "Affine":
                preview_decrypted_text, _ = self.current_module.decrypt_affine(self.current_module.model, self.current_module.tokenizer, self.current_module.max_len, input_text) # Verwende current_module
            self.preview_text.config(state=tk.NORMAL)
            self.preview_text.delete("1.0", tk.END)
            self.preview_text.insert("1.0", preview_decrypted_text)
            self.preview_text.config(state=tk.DISABLED)
        except Exception as e:
            self.preview_text.config(state=tk.NORMAL)
            self.preview_text.delete("1.0", tk.END)
            self.preview_text.insert("1.0", "Vorschau Fehler")
            self.preview_text.config(state=tk.DISABLED)


    def decrypt(self):
        if self.current_module.model is None: # Verwende current_module
            messagebox.showerror("Fehler", "Kein Modell geladen. Erstellen oder laden Sie zuerst ein Modell.")
            return

        input_text = self.input_text.get("1.0", tk.END).strip().lower()
        if not input_text:
            messagebox.showinfo("Info", "Bitte Text zum Entschlüsseln eingeben.")
            return

        try:
            if self.cipher_type == "Caesar":
                decrypted_text, key = self.current_module.decrypt_caesar(self.current_module.model, self.current_module.tokenizer, self.current_module.max_len, input_text) # Verwende current_module
                self.set_status(f"Entschlüsselt. Shift: {key}")
                self.key_display_label.config(text=f"Erkannter Shift-Wert: {key}")
            elif self.cipher_type == "Affine":
                decrypted_text, key = self.current_module.decrypt_affine(self.current_module.model, self.current_module.tokenizer, self.current_module.max_len, input_text) # Verwende current_module
                self.set_status(f"Entschlüsselt. Schlüssel: a={key[0]}, b={key[1]}")
                self.key_display_label.config(text=f"Erkannter Schlüssel: a={key[0]}, b={key[1]}")

            self.update_preview()

        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Entschlüsseln des Texts: {e}")
            self.set_status("Fehler")
            self.key_display_label.config(text="")
            self.preview_text.config(state=tk.NORMAL)
            self.preview_text.delete("1.0", tk.END)
            self.preview_text.config(state=tk.DISABLED)

    def paste_text(self, event):
        try:
            text = self.master.clipboard_get()
            current_widget = self.master.focus_get()
            if isinstance(current_widget, tk.Text) or isinstance(current_widget, scrolledtext.ScrolledText):
                current_widget.insert(tk.INSERT, text)
        except tk.TclError:
            pass
        return "break"

    def clear_training_texts(self):
        pass # Implementierung je nach Bedarf

    def clear_decryption(self):
        self.input_text.delete("1.0", tk.END)
        self.key_display_label.config(text="")
        self.preview_text.config(state=tk.NORMAL)
        self.preview_text.delete("1.0", tk.END)
        self.preview_text.config(state=tk.DISABLED)

    def clear_encryption(self):
        self.encryption_input_text.delete("1.0", tk.END)
        self.encrypted_text_display.config(state=tk.NORMAL)
        self.encrypted_text_display.delete("1.0", tk.END)
        self.encrypted_text_display.config(state=tk.DISABLED)
        self.encryption_key_display_label.config(text="")

    def edit_training_texts(self):
        pass # Implementierung je nach Bedarf

    def save_model(self):
        if self.current_module.model is None: # Verwende current_module
            messagebox.showerror("Fehler", "Kein Modell geladen. Erstellen oder laden Sie zuerst ein Modell.")
            return

        try:
            self.set_status("Modell wird gespeichert...")
            model_file_path = f'{self.cipher_type.lower()}_model.h5' # Dynamischer Dateiname
            self.current_module.save_model_and_params(self.current_module.model, self.current_module.tokenizer, self.current_module.max_len, model_file_path) # Verwende current_module
            self.set_status("Modell gespeichert.")
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Speichern des Modells: {e}")
            self.set_status("Fehler")

    def load_from_module(self):
        self.model = self.current_module.model # Verwende current_module
        self.training_texts = self.current_module.training_texts # Verwende current_module
        self.data_array = self.current_module.data_array # Verwende current_module
        self.labels_array = self.current_module.labels_array # Verwende current_module
        self.tokenizer = self.current_module.tokenizer # Verwende current_module
        self.max_len = self.current_module.max_len # Verwende current_module

    def encrypt_text(self):
        plaintext = self.encryption_input_text.get("1.0", tk.END)
        if not plaintext:
            messagebox.showinfo("Info", "Bitte Klartext zum Verschlüsseln eingeben.")
            return

        plaintext = self.current_module.replace_umlauts(plaintext).strip().lower() # Verwende current_module für Umlautersetzung

        try:
            if self.cipher_type == "Caesar":
                shift = int(self.shift_entry.get()) # Hole Caesar Shift Wert aus Entry
                if not 0 <= shift <= 25:
                    messagebox.showerror("Fehler", "Shift-Wert muss zwischen 0 und 25 liegen.")
                    return
                encrypted_text = self.current_module.encrypt_caesar(plaintext, shift) # Verwende current_module
                self.encryption_key_display_label.config(text=f"Verwendeter Shift-Wert: {shift}") # Anzeige für Caesar
                self.set_status(f"Verschlüsselt mit Shift: {shift}")

            elif self.cipher_type == "Affine":
                a = int(self.a_entry.get()) # Hole Affine 'a' und 'b' Werte aus Entry
                b = int(self.b_entry.get())
                valid_a_values = self.affine_module.VALID_A # Nutze affine_module für VALID_A
                if a not in valid_a_values:
                    messagebox.showerror("Fehler", f"Ungültiger Wert für 'a'. Gültige Werte sind: {valid_a_values}")
                    return
                if not 0 <= b <= 25:
                    messagebox.showerror("Fehler", "Wert für 'b' muss zwischen 0 und 25 liegen.")
                    return
                encrypted_text = self.current_module.affine_encrypt(plaintext, a, b) # Verwende current_module
                self.encryption_key_display_label.config(text=f"Verwendeter Schlüssel: a={a}, b={b}") # Anzeige für Affine
                self.set_status(f"Verschlüsselt mit Schlüssel: a={a}, b={b}")

            self.encrypted_text_display.config(state=tk.NORMAL)
            self.encrypted_text_display.delete("1.0", tk.END)
            self.encrypted_text_display.insert("1.0", encrypted_text)
            self.encrypted_text_display.config(state=tk.DISABLED)

        except ValueError:
            messagebox.showerror("Fehler", "Ungültige Zahleneingabe.")
            self.set_status("Fehler")
            self.encrypted_text_display.config(state=tk.NORMAL)
            self.encrypted_text_display.delete("1.0", tk.END)
            self.encrypted_text_display.config(state=tk.DISABLED)
            self.encryption_key_display_label.config(text="")

    def copy_encrypted_text(self):
        encrypted_text = self.encrypted_text_display.get("1.0", tk.END).strip()
        if encrypted_text:
            self.master.clipboard_clear()
            self.master.clipboard_append(encrypted_text)
            self.master.update()
            self.set_status("Verschlüsselter Text in die Zwischenablage kopiert.")
        else:
            messagebox.showinfo("Info", "Kein verschlüsselter Text zum Kopieren vorhanden.")


if __name__ == "__main__":
    root = ttkb.Window(themename=current_theme)
    ui = CipherUI(root) # Verwende CipherUI
    root.mainloop()