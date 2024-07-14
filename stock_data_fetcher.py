from datetime import datetime

import pandas as pd
import requests


class StockDataFetcher:
    def __init__(self, symbol):
        self.symbol = symbol

    def _get_time(self):
        return int(datetime.now().timestamp())

    def _get_data(self):
        filename = f'{self.symbol}.csv'
        start_date = 0
        end_date = self._get_time()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.99 Safari/537.36'}
        url = f"https://query1.finance.yahoo.com/v7/finance/download/{self.symbol}?period1={start_date}&period2={end_date}&interval=1d&events=history"
        response = requests.get(url, headers=headers)
        with open(filename, 'wb') as handle:
            for block in response.iter_content(1024):
                handle.write(block)
        data = pd.read_csv(filename, parse_dates=['Date'])
        data.set_index('Date', inplace=True)
        return data
\