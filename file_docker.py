# file_docker.py
import os
import shutil
from tkinter import Tk, filedialog

def load_documents_from(folder):
    docs = []
    for filename in os.listdir(folder):
        full = os.path.join(folder, filename)
        if os.path.isfile(full):
            docs.append({"name": filename, "path": full})
    return docs

def choose_files_popup(target_folder="presentation_files"):
    os.makedirs(target_folder, exist_ok=True)

    # Create popup
    root = Tk()
    root.title("Select Presentation Files")
    root.geometry("400x150")
    root.attributes("-topmost", True)

    def select_files():
        paths = filedialog.askopenfilenames(
            title="Choose files for presentation"
        )
        if paths:
            # Clear old files
            for f in os.listdir(target_folder):
                os.remove(os.path.join(target_folder, f))

            # Copy new files
            for p in paths:
                filename = os.path.basename(p)
                shutil.copy(p, os.path.join(target_folder, filename))

            root.destroy()

    import tkinter as tk
    btn = tk.Button(root, text="Select Files", font=("Segoe UI", 14),
                    command=select_files)
    btn.pack(expand=True)

    root.mainloop()

    return load_documents_from(target_folder)
