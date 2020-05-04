import requests

url = 'http://0.0.0.0:5000/clusterize'
data = {
    'sentences': ['Quantum physics is quite important in science nowadays',
                  'Software engineering is hotter and hotter topic in the silicon valley',
                  'Investing in stocks and trading with them are not that easy',
                  'FOREX is the stock market for trading currencies',
                  'Warren Buffet is famous for making good investments. He knows stock markets'
                  ],
    'n_clusters': 2
}

if __name__ == '__main__':
    r = requests.post(url, json=data)
    print(r.text)
