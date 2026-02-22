import tkinter as tk

class BreakWindow:
    def __init__(self):
        self.root = None
        self.isActive = False

    def show(self):
        if self.root is not None:
            return

        self.root = tk.Tk()
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-topmost", True)
        self.root.protocol("WM_DELETE_WINDOW", lambda: None)
        self.isActive = True

        tk.Label(self.root,
                text="Time for a break!",
                font=("Helvetica", 40)).pack(expand=True)

    def close(self):
        if self.root:
            self.root.destroy()
            self.root = None
            self.isActive = False

    def active(self):
        return self.isActive