from bs4 import BeautifulSoup
import requests


page = requests.get("https://beautiful-soup-4.readthedocs.io/en/latest/").content

soup = BeautifulSoup(page,'html.parser')
print(soup.prettify())


def page_html_into_file_writer():
    with open("html_parser.txt", "r", encoding = "utf-8") as f:
         row = f.read()
         if not row:
             with open("html_parser.txt", "w", encoding = "utf-8") as f:
                 f.write(soup.prettify())
         else:
             print("There is already some content!")


def find_data_div_wise():
    div_data = soup.find_all('div')
    print(div_data)

def find_nested_elements():
    for link in soup.find_all('a'):
        print(link.get('href',"No-Link"))


def finding_public_ip():
    ip = requests.get('https://api64.ipify.org?format=json')
    print(ip.json())


if __name__ == '__main__':
    finding_public_ip()
    find_data_div_wise()
    find_nested_elements()