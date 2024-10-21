# Town Coordinates
town_coordinates = {
    'ANG MO KIO': (1.375, 103.848), 
    'BEDOK': (1.324, 103.929), 
    'BISHAN': (1.358, 103.848), 
    'BUKIT BATOK': (1.366, 103.763),
    'BUKIT MERAH': (1.274, 103.822),
    'BUKIT TIMAH': (1.333, 103.776), 
    'CENTRAL AREA': (1.299, 103.849), 
    'CHOA CHU KANG': (1.389, 103.749), 
    'CLEMENTI': (1.317, 103.764),    
    'GEYLANG': (1.313, 103.867), 
    'HOUGANG': (1.367, 103.891), 
    'JURONG EAST': (1.334, 103.733), 
    'JURONG WEST': (1.350, 103.706),
    'KALLANG/WHAMPOA': (1.316, 103.857), 
    'MARINE PARADE': (1.304, 103.901), 
    'QUEENSTOWN': (1.290, 103.798), 
    'SENGKANG': (1.375, 103.895),
    'SERANGOON': (1.358, 103.870), 
    'TAMPINES': (1.351, 103.940), 
    'TOA PAYOH': (1.334, 103.846), 
    'WOODLANDS': (1.438, 103.786), 
    'YISHUN': (1.426, 103.836),
    'LIM CHU KANG': (1.437, 103.709), 
    'SEMBAWANG': (1.450, 103.834), 
    'BUKIT PANJANG': (1.377, 103.769), 
    'PASIR RIS': (1.372, 103.948),
    'PUNGGOL': (1.403, 103.903)
}

# towns
towns = list(town_coordinates.keys())

# Flat Types
flat_types = ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI GENERATION']

# Flat Models
flat_models = ['Improved', 'New Generation', 'Model A', 'Standard', 'Simplified',
               'Model A-Maisonette', 'Apartment', 'Maisonette', 'Terrace',
               '2-Room', 'Improved-Maisonette', 'Multi Generation',
               'Premium Apartment', 'Adjoined flat', 'Premium Maisonette',
               'Model A2', 'DBSS', 'Type S1', 'Type S2', 'Premium Apartment Loft',
               '3Gen']

# Final input features X
expected_columns = ['floor_area_sqm', 'nearest_supermarket_distance', 'nearest_school_distance', 'nearest_mrt_distance', 'nearest_hawkers_distance', 'cbd_distance', 'year_of_sale', 'calculated_remaining_lease', 'storey_median', 'town_BEDOK', 'town_BISHAN', 'town_BUKIT BATOK', 'town_BUKIT MERAH', 'town_BUKIT PANJANG', 'town_BUKIT TIMAH', 'town_CENTRAL AREA', 'town_CHOA CHU KANG', 'town_CLEMENTI', 'town_GEYLANG', 'town_HOUGANG', 'town_JURONG EAST', 'town_JURONG WEST', 'town_KALLANG/WHAMPOA', 'town_LIM CHU KANG', 'town_MARINE PARADE', 'town_PASIR RIS', 'town_PUNGGOL', 'town_QUEENSTOWN', 'town_SEMBAWANG', 'town_SENGKANG', 'town_SERANGOON', 'town_TAMPINES', 'town_TOA PAYOH', 'town_WOODLANDS', 'town_YISHUN', 'flat_type_2 ROOM', 'flat_type_3 ROOM', 'flat_type_4 ROOM', 'flat_type_5 ROOM', 'flat_type_EXECUTIVE', 'flat_type_MULTI GENERATION', 'flat_model_3Gen', 'flat_model_Adjoined flat', 'flat_model_Apartment', 'flat_model_DBSS', 'flat_model_Improved', 'flat_model_Improved-Maisonette', 'flat_model_Maisonette', 'flat_model_Model A', 'flat_model_Model A-Maisonette', 'flat_model_Model A2', 'flat_model_Multi Generation', 'flat_model_New Generation', 'flat_model_Premium Apartment', 'flat_model_Premium Apartment Loft', 'flat_model_Premium Maisonette', 'flat_model_Simplified', 'flat_model_Standard', 'flat_model_Terrace', 'flat_model_Type S1', 'flat_model_Type S2']