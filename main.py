from src.execute import execute_command
import yaml

if __name__ == "__main__":
    # tickers = ['BTC', 'ETH', 'USDT', 'SOL', 'BNB', 'DOGE', 'XRP', 'USDC', 'ADA', 'TRX']
    with open('..\\config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    key = config['alpha_vantage']['av_key']
    TOKEN = config['telegram']['bot_key']

    chat_id = config['telegram']['group_chat_id']
    ticker = 'ETH'
    execute_command(key, ticker, TOKEN, chat_id)

    chat_id = config['telegram']['btc_chat_id']
    ticker = 'BTC'
    execute_command(key, ticker, TOKEN, chat_id)