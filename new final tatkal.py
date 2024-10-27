from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import base64
from PIL import Image
import pytesseract
from PIL import ImageEnhance

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path if needed

# Replace these variables with your actual username and password
username = "your_username"
password = "your_password"

while True:  # Start an infinite loop
    # Set up the web driver (make sure to specify the correct path to your driver)
    driver = webdriver.Chrome()  # or webdriver.Firefox() for Firefox

    try:
        # Open the browser in full-screen mode
        driver.maximize_window()

        # Open the target URL
        driver.get("https://www.irctc.co.in/nget/train-search")  # Replace with the actual URL

        # Wait for the login button to be clickable
        try:
            login_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CLASS_NAME, "search_btn"))
            )
            login_button.click()
        except Exception as e:
            print(f"Error clicking the login button: {e}")
            driver.quit()
            continue  # Start the loop again

        # Wait for the username field to be present using formcontrolname
        try:
            username_field = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[formcontrolname='userid']"))
            )
            username_field.send_keys(username)
        except Exception as e:
            print(f"Error finding the username field: {e}")
            driver.quit()
            continue  # Start the loop again

        # Wait for the password field to be present using formcontrolname
        try:
            password_field = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[formcontrolname='password']"))
            )
            password_field.send_keys(password)
        except Exception as e:
            print(f"Error finding the password field: {e}")
            driver.quit()
            continue  # Start the loop again

        # Wait for the captcha image to be present
        try:
            captcha_img = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "img.captcha-img"))
            )
            captcha_src = captcha_img.get_attribute("src")

            # Extract the Base64 data from the src
            base64_data = captcha_src.split(",")[1]  # Get the part after "data:image/jpg;base64,"
            
            # Decode the Base64 data
            image_data = base64.b64decode(base64_data)

            # Save the image as captcha.jpg
            with open("captcha.jpg", "wb") as f:
                f.write(image_data)

            print("Captcha image saved as captcha.jpg")

            # Perform OCR on the image
            image = Image.open("captcha.jpg")
            # Convert to grayscale
            image = image.convert("L")

            # Optionally enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)  # Adjust the factor as needed
            extracted_text = pytesseract.image_to_string(image)

            # Print the extracted text
            print("Extracted Text from Captcha:", extracted_text)

            # Check if extracted_text is empty
            if not extracted_text.strip():  # If extracted_text is empty or only whitespace
                print("Extracted text is empty. Restarting the process...")
                driver.quit()
                continue  # Start the loop again

        except Exception as e:
            print(f"Error finding or saving the captcha image: {e}")
            driver.quit()
            continue  # Start the loop again

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    finally:
        # Wait a moment to see the result (optional)
        time.sleep(500)

        # Close the driver
        driver.quit()