import tkinter as tk
from tkinter import filedialog, messagebox
from detection import tflite_detect_images, draw_bounding_boxes

def browse_file(var):
    filename = filedialog.askopenfilename()
    var.set(filename)

def browse_directory(var):
    directory = filedialog.askdirectory()
    var.set(directory)

def start_detection():
    try:
        model_path = model_var.get()
        img_path = img_var.get()
        lbl_path = label_var.get()
        save_path = output_image_var.get()
        xml_save_path = output_xml_var.get()
        min_conf = float(conf_var.get())
        num_test_images = int(num_var.get())
        display_labels = display_var.get()
        
        tflite_detect_images(model_path, img_path, lbl_path, min_conf, num_test_images, save_path, xml_save_path)
        
        if display_labels:
            draw_bounding_boxes(img_path, xml_save_path, show_in_gui=True)
        
        messagebox.showinfo("Success", "Detection completed successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

root = tk.Tk()
root.title("TFLite Object Detection GUI")

# Model Path
tk.Label(root, text="Model Path:").grid(row=0, column=0, sticky=tk.W)
model_var = tk.StringVar()
tk.Entry(root, textvariable=model_var, width=50).grid(row=0, column=1)
tk.Button(root, text="Browse", command=lambda: browse_file(model_var)).grid(row=0, column=2)

# Image Folder
tk.Label(root, text="Image Folder:").grid(row=1, column=0, sticky=tk.W)
img_var = tk.StringVar()
tk.Entry(root, textvariable=img_var, width=50).grid(row=1, column=1)
tk.Button(root, text="Browse", command=lambda: browse_directory(img_var)).grid(row=1, column=2)

# Label Path
tk.Label(root, text="Label Path:").grid(row=2, column=0, sticky=tk.W)
label_var = tk.StringVar()
tk.Entry(root, textvariable=label_var, width=50).grid(row=2, column=1)
tk.Button(root, text="Browse", command=lambda: browse_file(label_var)).grid(row=2, column=2)

# Output Image Folder
tk.Label(root, text="Output Image Folder:").grid(row=3, column=0, sticky=tk.W)
output_image_var = tk.StringVar()
tk.Entry(root, textvariable=output_image_var, width=50).grid(row=3, column=1)
tk.Button(root, text="Browse", command=lambda: browse_directory(output_image_var)).grid(row=3, column=2)

# Output XML Folder
tk.Label(root, text="Output XML Folder:").grid(row=4, column=0, sticky=tk.W)
output_xml_var = tk.StringVar()
tk.Entry(root, textvariable=output_xml_var, width=50).grid(row=4, column=1)
tk.Button(root, text="Browse", command=lambda: browse_directory(output_xml_var)).grid(row=4, column=2)

# Confidence Threshold
tk.Label(root, text="Min Confidence Threshold:").grid(row=5, column=0, sticky=tk.W)
conf_var = tk.StringVar(value="0.5")
tk.Entry(root, textvariable=conf_var).grid(row=5, column=1)

# Number of Images to Test
tk.Label(root, text="Number of Images to Test:").grid(row=6, column=0, sticky=tk.W)
num_var = tk.StringVar(value="10")
tk.Entry(root, textvariable=num_var).grid(row=6, column=1)

# Display Labeled Images Checkbox
display_var = tk.BooleanVar()
tk.Checkbutton(root, text="Display Labeled Images", variable=display_var).grid(row=7, column=1)

# Start Button
tk.Button(root, text="Start Detection", command=start_detection).grid(row=8, column=1, pady=10)

root.mainloop()
