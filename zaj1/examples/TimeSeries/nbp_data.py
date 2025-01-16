import json

import requests

def download_from_nbp():
    url_pattern = 'https://api.nbp.pl/api/exchangerates/rates/a/eur/{year}-01-01/{year}-12-31/?format=json'

    data = None
    for year in range(2010, 2025):
        response = requests.get(url_pattern.format(year=year))
        if data is None:
            data = response.json()
        else:
            data['rates'].extend(response.json()['rates'])
    with open('../../datasets/nbp_euro.json', 'w') as fp:
        json.dump(data, fp)

def from_nbp_json_to_series():
    rates = []
    with open('../../datasets/nbp_euro.json', 'r') as fp:
        data = json.load(fp)['rates']
    for record in data:
        rates.append(record['mid'])
    return rates

def prepare_nbp_dataset(rates):
    #TODO
    pass

if __name__ == '__main__':
    #download_from_nbp()
    rates = from_nbp_json_to_series()
    print(len(rates))
    prepare_nbp_dataset(rates)