from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import base64
from PIL import Image, ImageEnhance
import pytesseract

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path if needed

# Replace these variables with your actual username and password
username = "AkhVen1"
password = "Arhaanke1@"

# Set up the web driver
driver = webdriver.Chrome()

try:
    # Open the browser in full-screen mode
    driver.maximize_window()
    driver.get("https://www.irctc.co.in/nget/train-search")  # Replace with the actual URL

    # Locate and click the login button
    login_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.CLASS_NAME, "search_btn"))
    )
    login_button.click()

    # Locate the username field and enter the username
    username_field = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "input[formcontrolname='userid']"))
    )
    username_field.send_keys(username)

    # Locate the password field and enter the password
    password_field = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "input[formcontrolname='password']"))
    )
    password_field.send_keys(password)

    # Locate and process the captcha image
    captcha_img = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "img.captcha-img"))
    )
    captcha_src = captcha_img.get_attribute("src")
    base64_data = captcha_src.split(",")[1]
    image_data = base64.b64decode(base64_data)

    with open("captcha.jpg", "wb") as f:
        f.write(image_data)

    # Perform OCR on the image
    image = Image.open("captcha.jpg").convert("L")
    image = ImageEnhance.Contrast(image).enhance(2.0)
    extracted_text = pytesseract.image_to_string(image).strip()

    if extracted_text:
        print("Extracted Text from Captcha:", extracted_text)
        captcha_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input[formcontrolname='captcha']"))
        )
        captcha_input.send_keys(extracted_text)

        # Input 'From' location
        from_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, 'origin')).find_element(By.TAG_NAME, 'input')
        )
        from_input.clear()
        from_input.send_keys('CSMT')
        from_input.send_keys(Keys.RETURN)

        # Input 'To' location
        to_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, 'destination')).find_element(By.TAG_NAME, 'input')
        )
        to_input.clear()
        to_input.send_keys('NGP')
        to_input.send_keys(Keys.RETURN)

        # Select 'AC 3 Tier' class
        dropdown_trigger = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, 'journeyClass'))
        )
        dropdown_trigger.click()
        ac_3_tier_option = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//li[contains(@aria-label, 'AC 3 Tier (3A)')]"))
        )
        ac_3_tier_option.click()

        # Select 'TATKAL' quota
        quota_dropdown = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, 'journeyQuota'))
        )
        quota_dropdown.click()
        tatkal_option = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//li[@aria-label='TATKAL']"))
        )
        tatkal_option.click()

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    driver.quit()
