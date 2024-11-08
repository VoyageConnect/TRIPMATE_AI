import os
import requests
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
import API_KEYS
from geopy.distance import geodesic

# Flask app initialization and CORS
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# API keys setup
OPENAI_API_KEY = API_KEYS.OPENAI_API_KEY
GOOGLE_PLACES_API_KEY = API_KEYS.GOOGLE_MAPS_API_KEY

# Directory and category files configuration
data_dir = "visitseoulData"
category_files = {
    "culturalTourism": ["cultural_tourism_with_location.txt", "attractions_museum_art_with_location.txt"],
    "historicalTourism": ["historical_tourism_with_location.txt", "attractions_historicalPlaces_with_location.txt",
                          "attractions_palace_palace_with_location.txt"],
    "shopping": ["shopping_with_location.txt", "attractions_shopping_with_location.txt"],
    "leisureSports": ["entertainments_with_location.txt"],
    "restaurant": ["restaurants_with_location.txt"],
    "natureTourism": ["nature_tourism_with_location.txt"]
}

# Initialize data structures
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)


def parse_location_data(content, source_file):
    """Parse location data from text content with source file tracking."""
    locations = []
    current_item = {}

    for line in content.split('\n'):
        if line.startswith('Title: '):
            if current_item:
                locations.append(current_item.copy())
            current_item = {
                'name': line[7:].strip(),
                'source_files': set([source_file]),
                'categories': set()
            }
        elif line.startswith('Description: '):
            current_item['description'] = line[12:].strip()
        elif line.startswith('Location: '):
            loc_data = line[10:].strip()
            try:
                lat = float(loc_data.split('Latitude = ')[1].split(',')[0])
                lon = float(loc_data.split('Longitude = ')[1])
                current_item['latitude'] = lat
                current_item['longitude'] = lon
            except:
                continue

    if current_item:
        locations.append(current_item)

    return locations


# Location data loading with category tracking
location_data = []
location_categories = {}
all_documents = []

# Load and process data
for category, file_names in category_files.items():
    if isinstance(file_names, str):
        file_names = [file_names]

    for file_name in file_names:
        file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(file_path):
            logger.warning(f"{file_name} file not found in {data_dir}. Skipping this category.")
            continue

        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            locations = parse_location_data(content, file_name)

            for loc in locations:
                name = loc['name']
                if name not in location_categories:
                    location_categories[name] = set()
                location_categories[name].add(category)

                existing_loc = next((x for x in location_data if x['name'] == name), None)
                if existing_loc:
                    existing_loc['source_files'].update(loc['source_files'])
                    existing_loc['categories'] = location_categories[name]
                else:
                    loc['categories'] = location_categories[name]
                    location_data.append(loc)

                doc_text = f"Name: {loc['name']}\nDescription: {loc.get('description', '')}\nCategories: {', '.join(loc['categories'])}"
                all_documents.append(Document(
                    page_content=doc_text,
                    metadata={
                        "name": loc['name'],
                        "latitude": loc['latitude'],
                        "longitude": loc['longitude'],
                        "categories": list(loc['categories'])
                    }
                ))

# Initialize OpenAI components
if all_documents:
    embedding_instance = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(documents=all_documents, embedding=embedding_instance)
    chat = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        temperature=0.7,
        max_tokens=200
    )
else:
    logger.warning("No documents were loaded. FAISS vectorstore initialization skipped.")


def get_category_weight(categories, user_preferences):
    """Calculate weight for a place based on its categories and user preferences."""
    if not categories:
        return 0

    total_weight = sum(user_preferences.get(category, 0) for category in categories)
    return total_weight / len(categories) if categories else 0


def get_place_id(lat, lon, place_name):
    """Get Google Place ID for a specific location."""
    try:
        url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={lat},{lon}&radius=100&keyword={place_name}&key={GOOGLE_PLACES_API_KEY}&language=ko"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'OK' and data.get('results'):
                for result in data['results']:
                    if result['name'].lower() == place_name.lower():
                        return result['place_id']
                return data['results'][0]['place_id']
    except Exception as e:
        logger.error(f"Error getting place ID: {str(e)}")
    return None


def enhance_description(place, user_preferences, distance_km):
    """Enhanced place description generation with better completion."""
    original_desc = place.get('description', '')
    categories = place.get('categories', set())

    high_preference_categories = [
        cat for cat in categories
        if cat in user_preferences and user_preferences[cat] > 0.5
    ]

    preference_context = "당신이 선호하는 " + ", ".join(high_preference_categories) if high_preference_categories else ""
    distance_text = f"{distance_km:.1f}km" if distance_km >= 1 else f"{int(distance_km * 1000)}m"

    category_context = f"이 장소는 {', '.join(categories)}에 속하는 곳으로"

    messages = [{
        "role": "system",
        "content": """당신은 서울의 관광 명소, 맛집, 문화 공간에 대해 해박한 지식을 가진 현지 가이드입니다.
        항상 완성된 문장으로 설명을 제공하며, 마지막 문장이 중간에 끊기지 않도록 합니다. 정확하지 않은 내용은 포함하지 않도록 합니다.
        장소의 특징과 매력을 사용자의 선호도를 고려하여 설명합니다."""
    }, {
        "role": "user",
        "content": f"""
        장소명: {place['name']}
        기존 설명: {original_desc}
        카테고리 정보: {category_context}
        현재 위치에서의 거리: {distance_text}
        사용자 선호: {preference_context}

        위 장소를 방문해야 하는 이유와 특징을 포함하여 완성된 두 문장으로 풍부하고 매력적으로 설명해주세요.
        카테고리와 관련된 특징을 반드시 포함하고, 문장이 중간에 끊기지 않도록 해주세요.
        """
    }]

    try:
        response = chat.generate([messages])
        return response.generations[0][0].text.strip()
    except Exception as e:
        logger.error(f"Error generating enhanced description: {str(e)}")
        return original_desc


def find_nearby_internal_places(lat, lon, radius_km=3):
    """Find nearby places from internal data."""
    nearby_places = []
    user_location = (lat, lon)

    for place in location_data:
        place_location = (place['latitude'], place['longitude'])
        distance = geodesic(user_location, place_location).kilometers

        if distance <= radius_km:
            place_copy = place.copy()
            place_copy['distance'] = distance
            nearby_places.append(place_copy)

    return sorted(nearby_places, key=lambda x: x['distance'])


def search_nearby_places(lat, lon):
    """Search for nearby places using Google Places API."""
    results = []
    try:
        url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={lat},{lon}&radius=3000&key={GOOGLE_PLACES_API_KEY}&language=ko"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'OK':
                for place in data['results']:
                    result = {
                        'name': place['name'],
                        'latitude': place['geometry']['location']['lat'],
                        'longitude': place['geometry']['location']['lng'],
                        'place_id': place.get('place_id'),
                        'categories': place.get('types', []),
                    }
                    results.append(result)
    except Exception as e:
        logger.error(f"Error searching nearby places: {str(e)}")

    return results


def search_google_place_info(place_id):
    """Fetch detailed information about a place from Google Places API."""
    try:
        url = f"https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&key={GOOGLE_PLACES_API_KEY}&language=ko"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'OK':
                result = data.get('result', {})
                description = result.get('formatted_address', '') + ' ' + result.get('name', '')
                categories = result.get('types', [])
                return {
                    'description': description,
                    'categories': categories
                }
    except Exception as e:
        logger.error(f"Error fetching place info from Google: {str(e)}")
    return None


def recommend_places(user_id, lat, lon):
    """Recommend places based on user preferences and location."""
    # User preferences fetching from DB can be added here; mock data for now
    user_preferences = {
        "culturalTourism": 0.7,
        "historicalTourism": 0.8,
        "shopping": 0.3,
        "leisureSports": 0.1,
        "restaurant": 0.0,
        "natureTourism": 0.0
    }

    # Find nearby internal places
    nearby_places = find_nearby_internal_places(lat, lon)

    # Process and weight places
    weighted_places = []
    for place in nearby_places:
        weight = get_category_weight(place['categories'], user_preferences)
        place['weight'] = weight
        weighted_places.append(place)

    # Sort by weight and distance
    sorted_places = sorted(weighted_places, key=lambda x: (-x['weight'], x['distance']))

    # Limit to top 5 recommendations
    top_recommendations = sorted_places[:5]

    # Enhance descriptions and find Google Place IDs
    final_recommendations = []
    for place in top_recommendations:
        distance_km = place['distance']

        # Check if description is missing and fetch from Google Places API if necessary
        if not place.get('description'):
            place_id = get_place_id(place['latitude'], place['longitude'], place['name'])
            if place_id:
                google_place_info = search_google_place_info(place_id)
                if google_place_info:
                    place['description'] = google_place_info.get('description', '')
                    place['categories'] = google_place_info.get('categories', [])
                else:
                    place['description'] = "정보를 찾을 수 없습니다."
            else:
                place['description'] = "정보를 찾을 수 없습니다."

        enhanced_description = enhance_description(place, user_preferences, distance_km)
        place_id = get_place_id(place['latitude'], place['longitude'], place['name'])

        # Determine the primary type based on highest weight category
        primary_type = max(place['categories'],
                           key=lambda x: user_preferences.get(x, 0)) if place['categories'] else "unknown"

        final_recommendations.append({
            "name": place['name'],
            "description": enhanced_description,
            "distance": distance_km,
            "place_id": place_id,
            "source": ", ".join(place.get('source_files', [])),
            "type": primary_type
        })

    return {"results": final_recommendations}


@app.route('/recommend', methods=['POST'])
def recommend():
    """Endpoint to get place recommendations."""
    data = request.get_json()
    user_id = data.get('user_id')
    latitude = data.get('latitude')
    longitude = data.get('longitude')

    # user_id = "3726289152"
    print("user_location:", latitude, longitude)
    # 홍대입구
    # latitude = 37.557434302
    # longitude = 126.926960224
    # 서울시청
    # latitude = 37.5664056
    # longitude =  126.9778222
    # 광화문
    # latitude = 37.571648599
    # longitude = 126.976372775
    # 강남역
    # latitude = 37.496486063
    # longitude = 127.028361548

    # 성수역
    # latitude = 37.544641605
    # longitude = 127.055896738

    # 혜화역
    # latitude = 37.582083337
    # longitude = 127.001914726

    # 이태원역
    # latitude = 37.534522948
    # longitude = 126.994243914

    # 공릉역
    # latitude = 37.625588720
    # longitude = 127.073025701

    recommendations = recommend_places(user_id, latitude, longitude)
    return jsonify(recommendations), 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)