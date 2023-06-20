"""
Download and extract data.
"""
import urllib.request
import zipfile



def download_data():
    """
    Downloads and extracts the data from the URL.
    """
    url = "https://raw.githubusercontent.com/ltroin/RelmaModelFS/main/"
    url += "a1_RestaurantReviews_HistoricDump.zip"

    extract_dir = "data/raw"

    zip_path, _ = urllib.request.urlretrieve(url)
    with zipfile.ZipFile(zip_path, "r") as file:
        file.extractall(extract_dir)


if __name__ == "__main__":
    download_data()
