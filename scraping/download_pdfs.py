from electricityforecasting.logger.logger import logging
from electricityforecasting.exception.exception import ElectricityForecastingException
import sys
import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

BASE_DIR = os.path.join("E:\\", "Electricity Demand Prediction")
DOWNLOAD_DIR = os.path.join(BASE_DIR, "artifacts", "downloaded_pdfs")

URL = "https://grid-india.in/en/reports/weekly-report"  
BROWSER_LOCATION = "C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe"
CHROMEDRIVER_PATH = r"C:\Users\MOKSHITA\Downloads\chromedriver-win64\chromedriver-win64\chromedriver.exe"

class PDFScraper:
    @staticmethod
    def setup_driver():
        try:
            options = Options()
            options.binary_location = BROWSER_LOCATION
            service = Service(executable_path=CHROMEDRIVER_PATH)
            driver = webdriver.Chrome(service=service, options=options)
            return driver
        except Exception as e:
            raise ElectricityForecastingException(e, sys)

    @staticmethod
    def select_all_dropdown(driver, wait):
        try:
            dropdown_control = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "div.my-select__control")))
            dropdown_control.click()
            time.sleep(1)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div[class*='my-select__menu']")))
            all_option = wait.until(EC.element_to_be_clickable((By.XPATH, "//div[contains(@class, 'my-select__option') and text()='ALL']")))
            all_option.click()
            time.sleep(2)
        except Exception as e:
            raise ElectricityForecastingException(e, sys)

    @staticmethod
    def scrape_pdf_links(driver, wait):
        try:
            all_pdf_links = set()
            page_number = 1

            while True:
                print(f"Scraping Page {page_number}...")
                time.sleep(2)

                wait.until(EC.presence_of_all_elements_located((By.XPATH, "//a[contains(@href, '.pdf')]")))
                pdf_links = driver.find_elements(By.XPATH, "//a[contains(@href, '.pdf')]")

                for link in pdf_links:
                    driver.execute_script("window.scrollBy(0, 52);")
                    time.sleep(1)
                    href = link.get_attribute("href")
                    if href and href.endswith(".pdf"):
                        all_pdf_links.add(href)

                try:
                    next_button = driver.find_element(By.XPATH, "//button[@aria-label='Next Page']")
                    if next_button.get_attribute("disabled") is not None:
                        break
                    next_button.click()
                    time.sleep(2)
                    page_number += 1
                    driver.execute_script("window.scrollTo(0, 100);")
                    time.sleep(1)
                except Exception as e:
                    raise ElectricityForecastingException(e, sys)

            return all_pdf_links
        except Exception as e:
            raise ElectricityForecastingException(e, sys)

    @staticmethod
    def download_pdfs(pdf_links):
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
        logging.info(f"Downloading {len(pdf_links)} PDFs to {DOWNLOAD_DIR}")

        for i, link in enumerate(pdf_links, 1):
            filename = os.path.join(DOWNLOAD_DIR, f"report_{i:03d}.pdf")
            try:
                response = requests.get(link)
                with open(filename, "wb") as f:
                    f.write(response.content)

            except Exception as e:
                raise ElectricityForecastingException(e,sys)

        logging.info("All downloads complete.")


if __name__ == "__main__":
    pdf_scraper = PDFScraper()
    try:
        driver = pdf_scraper.setup_driver()
        wait = WebDriverWait(driver, 10)

        driver.get(URL)
        time.sleep(2)

        pdf_scraper.select_all_dropdown(driver, wait)
        all_pdf_links = pdf_scraper.scrape_pdf_links(driver, wait)
        logging.info(f"Total PDFs found: {len(all_pdf_links)}")
        if not all_pdf_links:
            logging.info("No PDF links found.")

        pdf_scraper.download_pdfs(all_pdf_links)

    except Exception as e:
        raise ElectricityForecastingException(e, sys)
    finally:
        if 'driver' in locals():
            driver.quit()

  