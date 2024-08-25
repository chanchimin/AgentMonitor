import requests
import backoff

class GoogleSearch:
    def __init__(self, api_key="***REMOVED***"):
        self.url = "https://google-search74.p.rapidapi.com/"
        self.headers = {
            "x-rapidapi-key": api_key,
            "x-rapidapi-host": "google-search74.p.rapidapi.com"
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
