"""
Download and extract data.
"""
import urllib.request
import zipfile


# unfortunately, this does not work, winerror 10054
def download_data():
    URL = "https://raw.githubusercontent.com/ltroin/RelmaModelFS/main/a1_RestaurantReviews_HistoricDump.zip"
    EXTRACT_DIR = "data/raw"

    zip_path, _ = urllib.request.urlretrieve(URL)
    with zipfile.ZipFile(zip_path, "r") as f:
        f.extractall(EXTRACT_DIR)


if __name__ == "__main__":
    download_data()
