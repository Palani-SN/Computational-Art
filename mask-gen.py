import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class MaskGenerator:
    def __init__(self, root):
        self.root = root
        self.root.title("Mask Generation Tool")

        # UI Frame for Controls
        self.frame = tk.Frame(root)
        self.frame.pack(fill=tk.X, padx=5, pady=5)

        self.btn_load = tk.Button(self.frame, text="Load Image", command=self.load_image)
        self.btn_load.pack(side=tk.LEFT, padx=5)

        self.btn_save = tk.Button(self.frame, text="Save Mask", command=self.save_mask)
        self.btn_save.pack(side=tk.RIGHT, padx=5)

        # Brush Size Input
        self.brush_size_label = tk.Label(self.frame, text="Brush Size:")
        self.brush_size_label.pack(side=tk.LEFT, padx=5)

        self.brush_size_var = tk.StringVar(value="5")
        self.brush_size_entry = tk.Entry(self.frame, textvariable=self.brush_size_var, width=5)
        self.brush_size_entry.pack(side=tk.LEFT)
        self.brush_size_entry.bind("<Return>", self.update_brush_size)

        # Undo Button
        self.btn_undo = tk.Button(self.frame, text="Undo", command=self.undo_last_stroke)
        self.btn_undo.pack(side=tk.LEFT, padx=5)

        # Canvas for Drawing
        self.canvas = tk.Canvas(root, cursor="cross", bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Image Variables
        self.image = None
        self.original_size = None  # Stores the original resolution
        self.resized_image = None
        self.tk_image = None
        self.mask = None
        self.resized_mask = None
        self.mask_history = []  # Stores previous mask states
        self.drawing = False
        self.brush_size = 5  # Default brush size
        self.last_x, self.last_y = None, None

        # Mouse Bindings
        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_mask)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.bmp")])
        if not file_path:
            return

        # Load Image
        self.image = cv2.imread(file_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        # Store original size
        self.original_size = self.image.shape[:2]  # (height, width)

        # Resize image to fit screen
        self.resized_image = self.resize_image_to_fit(self.image, max_width=800, max_height=600)

        # Get new dimensions
        self.height, self.width = self.resized_image.shape[:2]

        # Initialize mask (black image)
        self.resized_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        self.mask_history = []  # Clear history on new image load

        # Update the canvas
        self.update_canvas()

    def resize_image_to_fit(self, img, max_width, max_height):
        """Resize image to fit within the given max dimensions while maintaining aspect ratio."""
        h, w = img.shape[:2]
        scale = min(max_width / w, max_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def update_canvas(self):
        if self.resized_image is not None:
            overlay = self.resized_image.copy()
            overlay[self.resized_mask == 255] = [255, 0, 0]  # Show mask in red

            # Convert to Tkinter format
            pil_image = Image.fromarray(overlay)
            self.tk_image = ImageTk.PhotoImage(pil_image)

            # Update Canvas
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
            self.canvas.config(width=self.width, height=self.height)

    def start_draw(self, event):
        self.drawing = True
        self.last_x, self.last_y = event.x, event.y
        # Save the current state of the mask before drawing
        self.mask_history.append(self.resized_mask.copy())

    def draw_mask(self, event):
        if self.drawing and self.resized_image is not None:
            x, y = event.x, event.y

            if 0 <= x < self.width and 0 <= y < self.height:
                # Draw a smooth line
                if self.last_x is not None and self.last_y is not None:
                    cv2.line(self.resized_mask, (self.last_x, self.last_y), (x, y), 255, self.brush_size)

                self.last_x, self.last_y = x, y
                self.update_canvas()

    def stop_draw(self, event):
        self.drawing = False
        self.last_x, self.last_y = None, None

    def update_brush_size(self, event=None):
        """Update brush size from input field."""
        try:
            size = int(self.brush_size_var.get())
            self.brush_size = max(2, min(50, size))  # Limit between 2 and 50
        except ValueError:
            self.brush_size_var.set(str(self.brush_size))  # Reset to valid value

    def undo_last_stroke(self):
        """Undo the last stroke by restoring the previous mask state."""
        if self.mask_history:
            self.resized_mask = self.mask_history.pop()
            self.update_canvas()

    def save_mask(self):
        if self.resized_mask is not None:
            save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                     filetypes=[("PNG Files", "*.png"),
                                                                ("JPEG Files", "*.jpg"),
                                                                ("Bitmap Files", "*.bmp")])
            if save_path:
                # Resize mask back to original resolution
                original_mask = cv2.resize(self.resized_mask, (self.original_size[1], self.original_size[0]), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(save_path, original_mask)
                print("Mask saved at original resolution:", save_path)

if __name__ == "__main__":
    root = tk.Tk()
    app = MaskGenerator(root)
    root.mainloop()
