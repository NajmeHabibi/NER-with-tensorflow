import pandas as pd
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
import time
from database import Document
from bs4 import BeautifulSoup
import requests
from selenium import webdriver


class NewsCrawler:
    start_url = "https://www.cbc.ca/news"
    categories = ['business', 'health', 'entertainment', 'science']

    def __init__(self):
        self.news_df = pd.DataFrame(columns=['title', 'text'])

    def iter_news(self, limit=None):
        counter = 0
        for category in self.categories:
            print(f"crawling {category} category on CBC")
            driver = webdriver.Chrome(executable_path='/usr/local/bin/chromedriver')
            driver.get(f"{self.start_url}/{category}")
            while True:
                try:
                    time.sleep(3)
                    load_more_button = driver.find_element(By.CLASS_NAME, "loadMore")
                    print("load more news")
                    driver.execute_script("arguments[0].click();", load_more_button)
                except NoSuchElementException:
                    print("Reached bottom of page")
                    break

            soup = BeautifulSoup(driver.page_source, 'html.parser')
            driver.close()
            main_section = soup.find('div', class_='contentArea')
            news_links = main_section.find_all('a', href=True)

            for link in news_links:
                if counter >= limit:
                    return

                link = link['href']
                if 'https' not in link:
                    html_news_page = requests.get(f"https://www.cbc.ca{link}")
                else:
                    html_news_page = requests.get(link)
                news_page = BeautifulSoup(html_news_page.text, 'lxml')
                title = news_page.find('h1', class_='detailHeadline')
                text = news_page.find('div', class_='story')
                if title is not None and text is not None:
                    yield Document(title=title.text, text=text.text)
                    counter += 1

