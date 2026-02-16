from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin



def connect_to_url():
    url = 'https://beautiful-soup-4.readthedocs.io/en/latest/'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    print(f" Successfully connected to {url}")

    return soup


def scrape_specific_code(soup):

    data = soup.select('div.highlight pre span.s2')
    if data:
        for item in data:
            print(item.get_text())
    else:
        print("No data found")


def scrape_table(soup):
    data = soup.select_one('div.wy-table-responsive table.docutils')
    if data:
        for item in data.find_all('tr'):
            print(item)
    else:
        print("No data found")



def usage_of_filter_func():

    txt = "line 1\n\nline 2\nline 3\nline 4\n\n line 5\nline 6"

    fil = filter(None, txt.splitlines())
    for i in fil:
        print(i.strip())


def covid_table():
    url='https://covid19dashboard.mohfw.gov.in/'
    res = requests.get(url)
    sp = BeautifulSoup(res.content, 'html.parser')

    data = sp.select('div.data-table table.statetable.table.table-striped tr.row1')

    for i in data:
        print(i.get_text())



if __name__ == '__main__':
    soup_obj = connect_to_url()
    # scrape_specific_code(soup_obj)
    # scrape_table(soup_obj)
    covid_table()
