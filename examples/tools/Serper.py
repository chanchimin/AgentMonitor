import jsonpickle
import requests
import backoff
import json

class SerperGoogleSearch:
    def __init__(self,):
        self.url = "https://google.serper.dev/search"
        self.headers = {
          'X-API-KEY': '6b907231b7a5840280cfb3292ebbbe93b2710fcd',
          'Content-Type': 'application/json'
        }

    @backoff.on_exception(
        backoff.expo,
        exception=(requests.exceptions.RequestException, requests.exceptions.HTTPError),
        max_tries=5,
        jitter=backoff.full_jitter
    )
    def search(self, query, date_filter="d", region="ar-es"):
        payload = json.dumps({"q": query})
        response = requests.request("POST", self.url, headers=self.headers, data=payload)

        if response.status_code == 200:
            return response.json()["organic"]
        else:
            response.raise_for_status()

if __name__ == '__main__':


    search_engine = SerperGoogleSearch()
    response = search_engine.search("please help me to write a python script to search google using serper api")
    print(response)
