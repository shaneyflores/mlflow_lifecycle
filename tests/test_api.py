import pytest
import requests

def test_api():
    
    # The URL of your local API
    url = "http://127.0.0.1:8000/predict"

    # Dummy wine data
    test_params = {
        "radius_mean": 2,
        "texture_mean": 3,
        "perimeter_mean": 4,
        "area_mean": 5,
        "smoothness_mean": 6,
        "compactness_mean": 7,
        "concavity_mean": 8,
        "concave points_mean": 9,
        "symmetry_mean": 10,
        "fractal_dimension_mean": 11,
        "radius_se": 12,
        "texture_se": 13,
        "perimeter_se": 14,
        "area_se": 15,
        "smoothness_se": 16,
        "compactness_se": 17,
        "concavity_se": 18,
        "concave points_se": 19,
        "symmetry_se": 20,
        "fractal_dimension_se": 21,
        "radius_worst": 22,
        "texture_worst": 23,
        "perimeter_worst": 24,
        "area_worst": 25,
        "smoothness_worst": 26,
        "compactness_worst": 27,
        "concavity_worst": 28,
        "concave points_worst": 29,
        "symmetry_worst": 30,
        "fractal_dimension_worst": 31
    }

    print("Sending request to API...")
    try:
        response = requests.post(url, params=test_params) 
        
        if response.status_code != 200:
            raise Exception(f"API request failed with status code {response.status_code}")  

        print("Response Body:", response.json())
        
    except Exception as e:
        print("Test failed:", e)