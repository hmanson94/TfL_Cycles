import wget
import json
import time
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

caps = DesiredCapabilities.CHROME
caps['goog:loggingPrefs'] = {'performance':'ALL'}
# driver=webdriver.Chrome(desired_capabilities=caps)
driver=webdriver.Chrome(executable_path='C:\\Users\\hmanson94\\Documents\\Python\\PwC Tech Test\\chromedriver.exe')
driver.get('https://cycling.data.tfl.gov.uk/')

def process_browser_log_entry(entry):
    response = json.loads(entry['message'])['message']
    return response

time.sleep(8)
browser_log = driver.get_log('performance')
events = [process_browser_log_entry(entry) for entry in browser_log]
events = [event for event in events if 'Network.response' in event['method']]

s3_urls = []
years_to_check = ['2019','2020','2021']

for event in events:
    for year in years_to_check:
        if 'response' in event['params']:
            if 'url' in event['params']['response']:
                if '.csv' in event['params']['response']['url']:
                    if year in event['params']['response']['url']:
                        s3_urls.append(event['params']['response']['url'])
                        
for url in s3_urls:
    wget.download(url, out = r'C:\\Users\\hmanson94\\Documents\\Python\\PwC Tech Test\\scraped_data')
    
print(s3_urls)