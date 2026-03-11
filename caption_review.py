import tkinter as tk
from tkinter import messagebox, scrolledtext
import json
import requests
from io import BytesIO
from PIL import Image, ImageTk
import threading
import os
import re

# --- CONFIGURATION ---
# --- CONFIGURATION ---
with open("./our-dataset-captions.json", 'r') as f:
    data = json.load(f)

IMAGE_URLS = [data[i]["url"] for i in data]
INITIAL_CAPTIONS = [data[i]["caption"] for i in data]


TEMP_FILE = "caption_cache_temp.json"
FINAL_FILE = "reviewed_captions.json"

class CaptionReviewer:
    def __init__(self, root, urls, captions):
        self.root = root
        self.urls = urls
        self.captions = captions
        self.current_index = 0
        self.raw_images = {} 
        
        self.load_temp_data()
        self.setup_ui()
        
        self.root.bind("<Configure>", self.on_window_resize)
        self.display_current()

    def setup_ui(self):
        self.root.title("Image Caption Reviewer")
        self.root.geometry("900x800")
        self.root.configure(bg="#f8f9fa")

        # Image Container
        self.img_label = tk.Label(self.root, text="Loading...", bg="#e9ecef")
        self.img_label.pack(pady=20, padx=20)

        # Caption Area
        tk.Label(self.root, text="Edit Caption:", font=("Segoe UI", 10, "bold"), bg="#f8f9fa").pack()
        
        # ScrolledText Widget
        self.text_area = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD, width=80, height=6, 
            font=("Segoe UI", 11), undo=True
        )
        self.text_area.pack(pady=10, padx=40)
        
        # Bind Ctrl+Backspace
        self.text_area.bind("<Control-BackSpace>", self.ctrl_backspace)

        # Controls
        btn_frame = tk.Frame(self.root, bg="#f8f9fa")
        btn_frame.pack(pady=20)

        tk.Button(btn_frame, text="❮ Previous", command=self.go_prev, width=12).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Next ❯", command=self.go_next, width=12).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Done & Save", command=self.save_and_exit, 
                  width=15, bg="#28a745", fg="white", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=30)

        self.status = tk.Label(self.root, text="", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def ctrl_backspace(self, event):
        """Custom logic to delete the word behind the cursor."""
        # Get the position of the insert cursor
        insert_pos = self.text_area.index(tk.INSERT)
        # Search backward for the start of a word
        # This looks for transitions between space and non-space characters
        start_pos = self.text_area.search(r'\W\w', insert_pos, backwards=True, regexp=True)
        
        if not start_pos:
            # If no word boundary found, delete to the beginning of the line
            start_pos = f"{insert_pos.split('.')[0]}.0"
        else:
            # Move index forward by 1 to keep the space/punctuation
            start_pos = self.text_area.index(f"{start_pos} + 1 chars")

        self.text_area.delete(start_pos, insert_pos)
        return "break" # Prevent the default backspace behavior

    def load_temp_data(self):
        if os.path.exists(TEMP_FILE):
            try:
                with open(TEMP_FILE, 'r') as f:
                    self.captions = json.load(f)
            except: pass

    def save_temp_data(self):
        # Update current caption in list before saving
        current_text = self.text_area.get("1.0", tk.END).strip()
        self.captions[self.current_index] = current_text
        with open(TEMP_FILE, 'w') as f:
            json.dump(self.captions, f)

    def on_window_resize(self, event):
        if event.widget == self.root:
            # Delay slightly to wait for window geometry to stabilize
            self.root.after(50, self.refresh_image_display)

    def refresh_image_display(self):
        url = self.urls[self.current_index]
        if url in self.raw_images:
            # Target width = 80% of window width
            win_width = self.root.winfo_width()
            target_width = int(win_width * 0.8)
            
            if target_width < 50: return # Prevent errors on tiny windows
            
            orig_img = self.raw_images[url]
            aspect_ratio = orig_img.height / orig_img.width
            target_height = int(target_width * aspect_ratio)
            
            # Use high-quality resizing
            resized_img = orig_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(resized_img)
            
            self.img_label.config(image=photo, text="")
            self.img_label.image = photo 

    def display_current(self):
        self.status.config(text=f"Progress: {self.current_index + 1} / {len(self.urls)}")
        self.text_area.delete("1.0", tk.END)
        self.text_area.insert(tk.END, self.captions[self.current_index])
        self.text_area.edit_reset() # Reset undo stack for the new text
        
        url = self.urls[self.current_index]
        if url in self.raw_images:
            self.refresh_image_display()
        else:
            self.img_label.config(image='', text="Downloading Image...")
            threading.Thread(target=self.fetch_image, args=(url,), daemon=True).start()

    def fetch_image(self, url):
        try:
            response = requests.get(url, timeout=10)
            img_data = Image.open(BytesIO(response.content))
            self.raw_images[url] = img_data
            self.root.after(0, self.refresh_image_display)
        except Exception as e:
            self.root.after(0, lambda: self.img_label.config(text=f"Error: {e}"))

    def go_next(self):
        if self.current_index < len(self.urls) - 1:
            self.save_temp_data()
            self.current_index += 1
            self.display_current()

    def go_prev(self):
        if self.current_index > 0:
            self.save_temp_data()
            self.current_index -= 1
            self.display_current()

    def save_and_exit(self):
        self.save_temp_data()
        output = [{"url": u, "caption": c} for u, c in zip(self.urls, self.captions)]
        with open(FINAL_FILE, 'w') as f:
            json.dump(output, f, indent=4)
        if os.path.exists(TEMP_FILE): os.remove(TEMP_FILE)
        messagebox.showinfo("Success", f"Data saved to {FINAL_FILE}")
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CaptionReviewer(root, IMAGE_URLS, INITIAL_CAPTIONS)
    root.mainloop()