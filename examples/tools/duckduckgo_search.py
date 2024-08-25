import requests
import backoff

class DuckDuckGoSearch:
    def __init__(self, api_key="e4e58032c6msh10621fb3e7bc2d8p135fa7jsn4c41242b17e2"):
        self.url = "https://duckduckgo8.p.rapidapi.com/"
        self.headers = {
            "x-rapidapi-key": api_key,
            "x-rapidapi-host": "duckduckgo8.p.rapidapi.com"
        }

    @backoff.on_exception(
        backoff.expo,
        exception=(requests.exceptions.RequestException, requests.exceptions.HTTPError),
        max_tries=5,
        jitter=backoff.full_jitter
    )
    def search(self, query, date_filter="d", region="ar-es"):
        querystring = {"q": query, }
        response = requests.get(self.url, headers=self.headers, params=querystring)

        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()
