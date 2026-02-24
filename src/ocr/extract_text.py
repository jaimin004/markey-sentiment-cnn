import easyocr
import os
from tqdm import tqdm
from src.utils.text_cleaner import clean_text   # <-- import cleaner

# Paths
image_folder = "data/raw_images"
output_folder = "data/processed_text"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Supported image extensions
image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

# Loop through all images
for image_file in tqdm(os.listdir(image_folder)):
    if image_file.lower().endswith(image_extensions):
        image_path = os.path.join(image_folder, image_file)

        try:
            # Perform OCR
            result = reader.readtext(image_path, detail=0)

            # Convert list of lines into a single string
            raw_text = " ".join(result)

            # Clean text
            cleaned_text = clean_text(raw_text)

            # Save to .txt file
            txt_file = os.path.join(
                output_folder,
                os.path.splitext(image_file)[0] + ".txt"
            )

            with open(txt_file, "w", encoding="utf-8") as f:
                f.write(cleaned_text)

        except Exception as e:
            print(f"Error processing {image_file}: {e}")

print("OCR extraction complete!")
