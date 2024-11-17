# Project Overview

This repository contains a script that extracts cryptocurrency price data for Bitcoin and Ethereum. It calculates various technical indicators, which are then used to generate buy/sell signals and identify overbought or oversold conditions. When a signal is detected, a Telegram bot sends a notification to a designated group.

---

## Table of Contents
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [License](#license)

---

## Installation and Setup

This project uses `Python v3.11`, and the required libraries are listed in `requirements.txt`. Do `pip install -r requirements.txt` to install in your environment.

## Usage

Put your `Alpha-Vantage API key`, `Telegram Bot Key`, and `Telegram Group Chat-ID` in `config.yaml`, and copy `config.yaml` to the parent folder (one folder before `trading-strategy`) to avoid exposing the IDs to the repo.

## License

The code is provided as it is, and licensed with [MIT License](https://opensource.org/license/mit).