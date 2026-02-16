from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import pandas as pd


def selenium_pagination():
    """
    This function is used to scrape the table using selenium!
    """
    driver = webdriver.Chrome()
    all_data = []
    headers = []

    for page in range(1, 3):  # Scraping first 2 pages
        driver.get(f'https://www.scrapethissite.com/pages/forms/?page_num={page}')
        time.sleep(2)

        all_tr_elements = driver.find_elements(By.CSS_SELECTOR, 'table.table tr')

        if not all_tr_elements:
            continue

        if not headers:
            header_cells = all_tr_elements[0].find_elements(By.TAG_NAME, 'th')  #using tag-name -> th
            headers = [th.text.strip() for th in header_cells]

        for tr in all_tr_elements[1:]:
            row_cells = tr.find_elements(By.TAG_NAME, 'td')  #using tag-name -> td

            if row_cells:
                current_row = [td.text.strip() for td in row_cells]
                all_data.append(current_row)

        print(f"Scraped {len(all_data)} rows")

    driver.quit()     #closing the driver (IMP)


    # making a dataframe  just to display better
    df = pd.DataFrame(all_data, columns=headers)
    print("\n--- Final Data ---")
    print(df)




if __name__ == '__main__':
    selenium_pagination()