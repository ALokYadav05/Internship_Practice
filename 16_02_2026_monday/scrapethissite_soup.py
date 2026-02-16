from bs4 import BeautifulSoup
import requests
import pandas as pd
import time


def connect_to_website():
    url = 'https://www.scrapethissite.com/pages/forms/'
    response = requests.get(url)

    soup = BeautifulSoup(response.content,'html.parser')
    print(f" Successfully connected to {url}")
    return soup


def scrape_table(soup):
    data = []
    heading=[]
    table = soup.select_one('table.table')

    row = table.find_all('tr')
    if row:
        for th in row[0].find_all('th'):
            heading.append(th.text.strip())

    for tr in row:
        row_cells = []
        for td in tr.find_all('td'):
            row_cells.append(td.get_text(strip=True))

        if row_cells:
            data.append(row_cells)

    ###################### Another-way #########################
    # headers = [th.get_text(strip=True) for th in table.find_all('th')]
    # print(headers)
    # rows = []
    # for row in table.select('tr')[1:]:
    #     cols = [td.get_text(strip=True) for td in row.find_all('td')]
    #     rows.append(cols)
    # df = pd.DataFrame(rows, columns=headers)
    # print(df.to_string(index=False))

    print(heading)
    print(data)



def pagination():
    base_url = "https://www.scrapethissite.com/pages/forms/"
    headers = {"User-Agent": "Mozilla/5.0"}
    all_rows = []
    col_names = []
    for page in range(1,6):
        url = f"{base_url}?page_num={page}"
        response = requests.get(url,headers=headers)
        soup = BeautifulSoup(response.text,"html.parser")
        table = soup.select_one("table.table")
        if table is None:
            continue
        if not col_names:
            col_names = [th.get_text(strip=True) for th in table.find_all("th")]
        rows = table.select("tr")[1:]
        for row in rows:
            cols = [td.get_text(strip=True) for td in row.select("td")]
            all_rows.append(cols)
    print(f"Total rows: {len(all_rows)}")
    print(f"Total columns: {len(col_names)}")
    df = pd.DataFrame(all_rows, columns=col_names)
    print(df)



def pagination2():
    all_data = []

    for page in range(1, 3):
        print(f"Scraping Page {page}...")

        url = f"https://www.scrapethissite.com/pages/forms/?page_num={page}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        table = soup.select_one('table.table')
        rows = table.find_all('tr')

        for tr in rows[1:]:
            cells = tr.find_all('td')
            if cells:
                row_data = [td.get_text(strip=True) for td in cells]
                all_data.append(row_data)


        time.sleep(1)


    headers = [th.get_text(strip=True) for th in rows[0].find_all('th')]
    df = pd.DataFrame(all_data, columns=headers)
    print(df.head())



if __name__ == '__main__':
    soup_obj = connect_to_website()
    # scrape_table(soup_obj)
    pagination()
    # pagination2()