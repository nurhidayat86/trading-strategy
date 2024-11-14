from function import get_crypto_signal, send_message
import time

def execute_command(key, tickers, TOKEN, chat_id):
    # Code to be measured
    for ticker in tickers:
        start_time = time.time()  # Start the timer
        series_crp = get_crypto_signal(key, ticker=ticker, market='USD')
        send_str = f"\n\n{ticker}:\n{series_crp}"
        end_time = time.time()  # End the timer
        execution_time = end_time - start_time
        print(execution_time)
        send_message(TOKEN, chat_id, send_str)





