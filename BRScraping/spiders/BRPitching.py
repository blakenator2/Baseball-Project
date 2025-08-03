import scrapy
import pandas as pd
from scrapy_splash import SplashRequest
from urllib.parse import quote

def make_file():
    file = pd.read_excel(r'C:\Users\Blake\Desktop\Pitching-change.xlsx')
    return file
def read_excel():
    df = pd.read_excel(r'C:\Users\Blake\Desktop\Pitching-change.xlsx')
    return df['playerID'].values.tolist()
def read_excel_year():
    df = pd.read_excel(r'C:\Users\Blake\Desktop\Pitching-change.xlsx')
    return df['yearID'].values.tolist()

class BrpitchingSpider(scrapy.Spider):
    file = make_file()
    name = "BRPitching"
    allowed_domains = []
    backup_interval = 100
    backup_path = r'C:\Users\Blake\Desktop\Pitching-backup.xlsx'

    def start_requests(self):
        for index, codes in enumerate(read_excel()):
            encoded = quote(codes)
            url = f"https://www.baseball-reference.com/players/{encoded[0]}/{encoded}.shtml"
            yield SplashRequest(url, callback=self.parse, meta={'code': codes, 'index': index})

    def parse(self, response):
        name = response.meta['code']
        index = response.meta['index']
        years = read_excel_year()
        print(name, index)
        while index < len(self.file) and name == self.file.loc[index, 'playerID']:
            year = years[index]
            real_name = (response.xpath('//*[@id="meta"]//h1/span/text()').get())
            age = (response.xpath(f'//*[@id="players_standard_pitching.{year}"]//td[@data-stat="age"]/text()').get())
            position = (response.xpath(f'//*[@id="players_advanced_batting.{year}"]//td[@data-stat="pos"]/text()').get())
            if age is None: age = (response.xpath(f'//*[@id="players_advanced_batting.{year}"]//td[@data-stat="age"]/text()').get())
            self.file.loc[index, 'Name'] = real_name
            self.file.loc[index, 'Age'] = age
            self.file.loc[index, 'Position'] = position
            index += 1

        if index > self.backup_interval:
            self.file.to_excel(self.backup_path)
            self.backup_interval += 100

        yield {
            'playerID': name,
        }

    def closed(self, reason):
        self.file.to_excel(r'C:\Users\Blake\Desktop\Done.xlsx', index=False)
