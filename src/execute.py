from src.function import get_crypto_signal, send_message
from src.function import Crypto
import time

def execute_command(key, ticker, TOKEN, chat_id):
    # Code to be measured
    start_time = time.time()  # Start the timer
    series_crp = get_crypto_signal(key, ticker=ticker, market='USD')
    send_str = f"\n\n{ticker}:\n{series_crp}"
    end_time = time.time()  # End the timer
    execution_time = end_time - start_time
    print(send_str)
    print(execution_time)
    if (series_crp.loc[["sell_signal", "buy_signal"]].sum() > 0) or (series_crp.loc["oversold_confirm"].sum() != 0):
        send_message(TOKEN, chat_id, send_str)

def crypto_stored_db(key, ticker):
    crypto = Crypto(key=key, ticker=ticker, market="USD")
    df = crypto.get_intraday()
    df = crypto.rename_column(df)
    print(df.sort_index(ascending=False).head())
    print(df.date.dt.strftime("%Y%m%d%H%M%S").astype('int64').max())

def crypto_info_to_telegram(config):
    # tickers = ['BTC', 'ETH', 'USDT', 'SOL', 'BNB', 'DOGE', 'XRP', 'USDC', 'ADA', 'TRX']
    key = config['alpha_vantage']['av_key']
    TOKEN = config['telegram']['bot_key']

    chat_id = config['telegram']['group_chat_id']
    ticker = 'ETH'
    execute_command(key, ticker, TOKEN, chat_id)

    chat_id = config['telegram']['btc_chat_id']
    ticker = 'BTC'
    execute_command(key, ticker, TOKEN, chat_id)

def crypto_to_db(config):
    key = config['alpha_vantage']['av_key']
    ticker = 'ETH'
    crypto_stored_db(key, ticker)






