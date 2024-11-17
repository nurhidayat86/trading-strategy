from src.execute import crypto_info_to_telegram, crypto_to_db
import yaml

if __name__ == "__main__":
    with open('..\\config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    crypto_info_to_telegram(config)