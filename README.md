# SteamSale
Data Mining and Warehousing — Final Project

This project predicts discount depth and value retention of PC games on Steam from game attributes (genre, developer, age, reviews), and analyzes sale-event effectiveness on player engagement using historical price and concurrent-player data.

## Environment Setup

Locate the `environment.yml` file in your directory then use mamba or conda to install it in your machine. Use the environment as your kernel when running the code to ensure consistent versions of the libraries are used.

```
mamba env create -f environment.yml -y
```

## Data Collection

Create a `.env` file at the project root with the following keys:

```
STEAM_API_KEY=<your steam web api key>
ISTHEREANYDEAL_API=<your itad api key>
```

Then open `notebooks/01_data_collection.ipynb` and run the stages in order. Each stage updates progress flags on the `app_list` table, so the pipeline is resumable — safe to interrupt and restart.

## Authors

Created by:
- TODO

Course: DMW – Data Mining and Warehousing
Institution: Asian Institute of Management
