import requests
from geopy.distance import geodesic

def get_lat_long(address):
    try:
        url = f"https://www.onemap.gov.sg/api/common/elastic/search?searchVal={address}&returnGeom=Y&getAddrDetails=Y&pageNum=1"
        response = requests.get(url)
        data = response.json()
        if data['found'] > 0:
            latitude = data['results'][0]['LATITUDE']
            longitude = data['results'][0]['LONGITUDE']
            return latitude, longitude
        else:
            return None, None
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error fetching data for {address}: {e}")

def cal_lease_remaining_years(lease_commence_date, year_of_sale):
    return 99 - (year_of_sale - lease_commence_date)

def get_address(block, street_name):
    return f"{block} {street_name}"
