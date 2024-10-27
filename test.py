from selenium import webdriver
import time

# Set up the Chrome WebDriver
driver = webdriver.Chrome()  # Update the path

try:
    # Open Google
    driver.get("https://www.google.com")

    # Find the search box using its name attribute value
    search_box = driver.find_element("name", "q")

    # Type 'car' in the search box
    search_box.send_keys("car")

    # Submit the search form
    search_box.submit()  # Alternatively, you could use search_box.send_keys(Keys.RETURN)

    # Wait for a few seconds to allow the results to load
    time.sleep(3)

    # Scroll down using JavaScript
    for _ in range(10):  # Adjust the range for more or fewer scrolls
        driver.execute_script("window.scrollBy(0, 500);")  # Scroll down 500 pixels
        time.sleep(0.5)  # Wait half a second between scrolls

finally:
    # Close the browser after a while
    time.sleep(5)  # Keep the browser open for 5 seconds
    driver.quit()