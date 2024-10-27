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
username = "AkhVen1"
password = "Arhaanke1@"

# Set up the web driver
driver = webdriver.Chrome()  # or webdriver.Firefox() for Firefox

try:
    while True:  # Loop until we get valid extracted text
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
            exit()

        # Wait for the username field to be present using formcontrolname
        try:
            username_field = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[formcontrolname='userid']"))
            )
            username_field.send_keys(username)
        except Exception as e:
            print(f"Error finding the username field: {e}")
            driver.quit()
            exit()

        # Wait for the password field to be present using formcontrolname
        try:
            password_field = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[formcontrolname='password']"))
            )
            password_field.send_keys(password)
        except Exception as e:
            print(f"Error finding the password field: {e}")
            driver.quit()
            exit()

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
            while True:
                image = Image.open("captcha.jpg")
                # Convert to grayscale
                image = image.convert("L")

                # Optionally enhance contrast
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(2.0)  # Adjust the factor as needed
                
                # Perform OCR
                extracted_text = pytesseract.image_to_string(image)

                # Check if extracted text is non-empty
                if extracted_text.strip():  # Check if not just whitespace
                    print("Extracted Text from Captcha:", extracted_text)

                    # Locate the captcha input field and enter the extracted text
                    try:
                        captcha_input = WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, "input[formcontrolname='captcha']"))
                        )
                        captcha_input.send_keys(extracted_text)  # Input the extracted captcha text
                        try:
                            # Wait for the input element to be present using its class
                            input_element = WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located((By.CLASS_NAME, "ng-tns-c57-8"))
        )

                            # Clear the input field (optional)
                            input_element.clear()

                            # Input the value "CSMT"
                            input_element.send_keys("C SHIVAJI MAH T - CSMT (MUMBAI)")
                        except Exception as e:
                            print(f"An error occurred: {e}")    
                        break  # Exit the inner loop if text is successfully entered
                    except Exception as e:
                        print(f"Error finding captcha input field: {e}")
                        driver.quit()
                        exit()
                else:
                    print("Extracted text is empty, retrying...")

            # Break the outer loop if captcha is successfully solved
            break  # Exit the outer loop after successful extraction and input

        except Exception as e:
            print(f"Error finding or saving the captcha image: {e}")

except Exception as e:
    print(f"An unexpected error occurred: {e}")

finally:
    # Wait a moment to see the result (optional)
    time.sleep(500)

    # Close the driver
    driver.quit()
