from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from PIL import ImageEnhance

# Open the image
image = Image.open("captcha.jpg")

# Convert to grayscale
image = image.convert("L")

# Optionally enhance contrast
enhancer = ImageEnhance.Contrast(image)
image = enhancer.enhance(2.0)  # Adjust the factor as needed

# Now perform OCR
extracted_text = pytesseract.image_to_string(image)
print("Extracted Text from Captcha:", extracted_text)
