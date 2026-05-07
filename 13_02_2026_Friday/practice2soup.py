from urllib.parse  import urljoin
from bs4 import BeautifulSoup
import requests

url = "https://beautiful-soup-4.readthedocs.io/en/latest/"

headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (HTML, like Gecko) Chrome/144.0.0.0 Safari/537.36'}

def basic_config():
    """
    This function demonstrate the basic configuration of Beautiful Soup
    :returns : object of Beautiful Soup
    """

    try:
        response = requests.get(url, headers).content
        soup = BeautifulSoup(response, 'html.parser')
        return soup
    except Exception as e:
        print(f" Unexpected error: {e}")


def find_all_urls(soup):
    """
    This  function is used to find all urls in Beautiful Soup
    """

    try:
        for link in soup.find_all('a'):
            href = link.get('href',"No-Link")
            full_url = urljoin(url,href)        # using this now i can extract relative paths like this -> #multi-valued-attributes, #id12

            print(full_url)
    except Exception as e:
        print(f" Unexpected error: {e}")


def css_selector(soup):
    """
    This function demonstrate how to extract data using CSS selector
    """

    try:
        data = soup.select_one("div.section#beautiful-soup-documentation")  # 1st-way   (div.div_name#id_name)  select_one() gives us first element
        # data = soup.find_all("div",id = "beautiful-soup-documentation")   #normal soup-way
        for d in data:
            print(d.get_text())

        # paragraphs = soup.select("#beautiful-soup-documentation p")   # 2nd-way
        # for p in paragraphs:
        #     # strip=True cleans up extra whitespace/newlines
        #     print(p.get_text(strip=True))

    except Exception as e:
        print(f" Unexpected error: {e}")





if __name__ == "__main__":
    soup_obj = basic_config()
    # find_all_urls(soup_obj)
    css_selector(soup_obj)


