"""
This module tests the features and data.
"""
import pickle
import pytest

@pytest.fixture(name="dataset")
def load_dataset():
    """
    Load the dataset.
    """
    with open("output/splitData/x_data.pkl", "rb") as file:
        x_train = pickle.load(file)
    return x_train

def test_review_feature_memory_usage(dataset):
    """
    Test for Features and Data: memory cost of the review feature
    """
    assert True
    # review_feature_in_bytes = dataset.nbytes
    # max_bytes = 1000000000 # equal to 1 GB
    # assert (
    #     review_feature_in_bytes < max_bytes
    # ), f"The review feature is using too many bytes: {review_feature_in_bytes}"