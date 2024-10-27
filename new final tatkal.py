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
                               # Allow some time for the page to load
                            time.sleep(3)

                            # Locate the "From" input field by its ID
                            from_input = driver.find_element(By.ID, 'origin').find_element(By.TAG_NAME, 'input')

                            # Clear the field and input "CSMT"
                            from_input.clear()
                            from_input.send_keys('CSMT')

                            # Allow some time for autocomplete suggestions to load if necessary
                            time.sleep(1)

                            # Press Enter to select the suggestion (if necessary)
                            from_input.send_keys(Keys.RETURN)

                            # Locate the "To" input field by its ID
                            to_input = driver.find_element(By.ID, 'destination').find_element(By.TAG_NAME, 'input')

                            # Clear the field and input "NGP"
                            to_input.clear()
                            to_input.send_keys('NGP')

                            # Allow some time for autocomplete suggestions to load if necessary
                            time.sleep(1)
                            # Wait for the dropdown to be clickable and click it to open
                            dropdown_trigger = WebDriverWait(driver, 10).until(
                                EC.element_to_be_clickable((By.ID, 'journeyClass'))
                            )
                            dropdown_trigger.click()

                            # Wait for the dropdown options to be visible
                            dropdown_options = WebDriverWait(driver, 10).until(
                                EC.visibility_of_element_located((By.CLASS_NAME, 'ui-dropdown-items'))
                            )

                            # Find the "AC 3 Tier" option and click it
                            ac_3_tier_option = driver.find_element(By.XPATH, "//li[contains(@aria-label, 'AC 3 Tier (3A)')]")
                            ac_3_tier_option.click()

                            # Optionally, you can verify if the selection was successful
                            selected_option = driver.find_element(By.XPATH, "//span[contains(@class, 'ui-dropdown-label')]")
                            assert "AC 3 Tier (3A)" in selected_option.text

                            dropdown_trigger = WebDriverWait(driver, 10).until(
                            EC.element_to_be_clickable((By.ID, 'journeyQuota'))
                            )
                            dropdown_trigger.click()

                            # Wait for the dropdown options to be visible
                            # Here, we assume that the options are in a list that can be located by a class name or some other identifier
                            # You may need to adjust the selector based on your actual HTML structure
                            tatkal_option = WebDriverWait(driver, 10).until(
                                EC.element_to_be_clickable((By.XPATH, "//li[contains(text(), 'TATKAL')]"))
                            )
                            # tatkal_option.click()
                            driver.execute_script("arguments[0].click();", tatkal_option)


                            # Optionally, you can verify if the selection was successful
                            selected_option = driver.find_element(By.XPATH, "//span[contains(@class, 'ui-dropdown-label')]")
                            assert "TATKAL" in selected_option.text

                        except Exception as e:
                            print(f"An error occurred in inputting from location and selecting third class: {e}")    
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
    time.sleep(50)

    # Close the driver
    # driver.quit()
