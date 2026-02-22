import subprocess
import os
import tkinter as tk
from tkinter import ttk

# --- Correct Python paths ---
VENV1_PYTHON = r"C:\Users\peter\OneDrive\Documents\GitHub\J_Viision_Hack\Combined_folder\ForPeter\.venv\Scripts\python.exe"       # educator
SCRIPT1_PATH = "realtime_interface.py"

VENV2_PYTHON = r"C:\Users\peter\OneDrive\Documents\GitHub\J_Viision_Hack\Combined_folder\ForPeter\.face_venv\Scripts\python.exe" # student
SCRIPT2_PATH = "face_detect.py" if os.name == 'nt' else "face_detect_mac.py"

def run_script_in_env(python_executable, script_path):
    print(f"--- Running {script_path} with {python_executable} ---")
    try:
        result = subprocess.run(
            [python_executable, script_path],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"Output of {script_path}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}:")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
    except FileNotFoundError:
        print(f"Error: Python executable or script not found.")

def launch():
    selection = dropdown.get()
    root.destroy()
    if selection == "Educator":
        run_script_in_env(VENV1_PYTHON, SCRIPT1_PATH)
    else:
        run_script_in_env(VENV2_PYTHON, SCRIPT2_PATH)

# --- Build popup ---
root = tk.Tk()
root.title("Select Role")
root.geometry("300x150")
root.resizable(False, False)

tk.Label(root, text="Are you an educator or student?", pady=10).pack()

dropdown = ttk.Combobox(root, values=["Educator", "Student"], state="readonly")
dropdown.set("Student")
dropdown.pack(pady=5)

tk.Button(root, text="Launch", command=launch, pady=5).pack(pady=10)

root.mainloop()
