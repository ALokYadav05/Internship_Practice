from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options



def driver_setup():
    chrome_option = Options()
    chrome_option.incognito = True
    chrome_option.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36")
    driver = webdriver.Chrome()
    return driver


def connect_to_website(driver):
    url = 'https://www.scrapethissite.com/pages/simple/'
    response = driver.get(url)
    # response.raise_for_status()
    return response


def open_and_save_to_file(driver):
    try:
         file_path = r"C:\Users\alok.yadav\Desktop\Intership_practice\17_02_2026_Tuesday\data.html"
         with open(file_path,"w", encoding="utf-8") as f:
             content = driver.page_source      # so this is how we extract whole-page content in one-go!
             f.write(content)
         print("Successfully opened and saved file")
    except Exception as e:
        print("Error while writing-content!",e)


def open_html():
    try:
        url1 = r"C:\Users\alok.yadav\Desktop\Intership_practice\17_02_2026_Tuesday\data.html"
        option = Options()
        option.add_argument("--headless")
        driver = webdriver.Chrome(options = option)
        driver.get(url1)
        d = driver.find_elements(By.CSS_SELECTOR,'div.country-info')[:3]
        for i in d:
          print(i.text,"\n\n")


    except Exception as e:
        print(f" unexpected error: {e}")






if __name__ =='__main__':
    # driver_obj = driver_setup()
    # connect_to_website(driver_obj)
    # open_and_save_to_file(driver_obj)
    open_html()
