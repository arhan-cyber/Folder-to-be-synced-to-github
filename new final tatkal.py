from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import base64


# Replace these variables with your actual username and password
username = "your_username"
password = "your_password"

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

    # Optionally, submit the form if there is a submit button
    # password_field.send_keys(Keys.RETURN)  # Uncomment if needed

except Exception as e:
    print(f"An unexpected error occurred: {e}")





finally:
    # Wait a moment to see the result (optional)
    time.sleep(500)

    # Close the driver
    driver.quit()
