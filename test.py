from binance.client import Client
client = Client()
klines = client.futures_klines(symbol='FARTCOINUSDT', interval='1h')
print(len(klines), klines[-1])
