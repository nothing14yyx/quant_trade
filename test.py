from coinmetrics.api_client import CoinMetricsClient
client = CoinMetricsClient()          # 不传 api_key == Community
cat = client.catalog_asset_metrics_v2(assets=['btc']).to_list()
all_metrics = [m['metric'] for m in cat[0]['metrics']]
print(len(all_metrics), all_metrics[:10])   # 147 ['AdrActCnt', 'AdrBal1Cnt', ...]
