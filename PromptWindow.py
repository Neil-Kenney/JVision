import tkinter as tk
from tkinter import simpledialog

class PromptWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Configuration")
        self.root.geometry("400x200")
        self.break_interval = None
        self.away_threshold = None
    
    def get_settings(self):
        """Prompt user for break interval and away threshold, return both values"""
        self.break_interval = simpledialog.askfloat(
            "Configuration",
            "How long before a break (minutes):",
            initialvalue=25.0,
            minvalue=1.0,
            maxvalue=60.0
        )
        
        if self.break_interval is None:  # User cancelled
            return None, None
        
        self.away_threshold = simpledialog.askfloat(
            "Configuration",
            "How long should your break last (minutes):",
            initialvalue=5.0,
            minvalue=1.0,
            maxvalue=10.0
        )
        
        self.root.destroy()
        return self.break_interval, self.away_threshold