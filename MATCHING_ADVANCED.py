import pymysql
from fiona.transform import transform
from flask import Flask, request, jsonify
from flasgger import Swagger, swag_from
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
import API_KEYS
from typing import Dict, List, Tuple, Union
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler
import time
from openai import OpenAI
from flask_cors import CORS
import TRIPMATE_DB as DB
import logging
import openai
from pymysql.cursors import DictCursor
import math
from threading import Lock
import redis

app = Flask(__name__)
Swagger(app)
# CORS(app)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

openai_client = OpenAI(api_key=API_KEYS.OPENAI_API_KEY)
executor = ThreadPoolExecutor(max_workers=5)


class User:
    def __init__(self, user_data: Dict):
        self.user_id = user_data['user_id']
        self.age = user_data['age']
        self.gender = user_data['gender']
        self.cultural_tourism = user_data['cultural_tourism']
        self.food = user_data['food']
        self.historical_tourism = user_data['historical_tourism']
        self.leisure_sports = user_data['leisure_sports']
        self.nature_tourism = user_data['nature_tourism']
        self.shopping = user_data['shopping']
        self.preferred_age = user_data['preferred_age']
        self.preferred_gender = user_data['preferred_gender']
        self.latitude = user_data.get('latitude', None)
        self.longitude = user_data.get('longitude', None)
        self.request_time = time.time()


class MatchingService:
    def __init__(self):
        self.preference_scaler = MinMaxScaler()
        self.waiting_users = {}
        self.matching_results = {}
        self.matching_lock = Lock()
        self.conn = None
        self.connect_to_db()
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)

    def connect_to_db(self):
        """데이터베이스 연결을 수행하는 메소드"""
        try:
            if self.conn and self.conn.open:
                self.conn.close()
            self.conn = pymysql.connect(**DB.DB_CONFIG, autocommit=False)
            logger.info("Database connection established successfully")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def check_connection(self):
        """데이터베이스 연결 상태를 확인하고 필요시 재연결하는 메소드"""
        try:
            if not self.conn or not self.conn.open:
                logger.warning("Database connection lost. Attempting to reconnect...")
                self.connect_to_db()
            else:
                # 간단한 쿼리로 연결 상태 확인
                with self.conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
        except Exception as e:
            logger.error(f"Database connection check failed: {e}")
            self.connect_to_db()

    def execute_with_retry(self, operation, *args, max_retries=3):
        """데이터베이스 작업을 재시도 로직과 함께 실행하는 메소드"""
        for attempt in range(max_retries):
            try:
                self.check_connection()
                result = operation(*args)
                self.conn.commit()
                return result
            except pymysql.Error as e:
                logger.error(f"Database operation failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(1 * (attempt + 1))  # 지수 백오프
                self.connect_to_db()
        raise Exception("Maximum retries exceeded")

    @staticmethod
    def transform_gender(gender: str) -> str:
        gender_map = {"남성": "MALE", "여성": "FEMALE"}
        return gender_map.get(gender, gender)

    @staticmethod
    def transform_age(age: Union[int, str]) -> str:
        if isinstance(age, str):
            return age

        try:
            age = int(age)
        except (ValueError, TypeError):
            logger.error(f"Invalid age value: {age}")
            return "UNKNOWN"

        age_group_map = {
            2: "TWENTIES",
            3: "THIRTIES",
            4: "FORTIES",
            5: "FIFTIES",
            6: "SIXTIES",
        }
        return age_group_map.get(age // 10, "UNKNOWN")

    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
            return float('inf')

        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371
        return c * r

    def calculate_preference_similarity(self, pref1: Dict[str, int], pref2: Dict[str, int]) -> float:
        keys = ['cultural_tourism', 'food', 'historical_tourism',
                'leisure_sports', 'nature_tourism', 'shopping']
        pref1_array = np.array([pref1[k] for k in keys]).reshape(1, -1)
        pref2_array = np.array([pref2[k] for k in keys]).reshape(1, -1)

        if not hasattr(self.preference_scaler, 'n_features_in_'):
            self.preference_scaler.fit(np.vstack((pref1_array, pref2_array)))

        normalized_pref1 = self.preference_scaler.transform(pref1_array).flatten()
        normalized_pref2 = self.preference_scaler.transform(pref2_array).flatten()
        return 1 - euclidean(normalized_pref1, normalized_pref2)

    def is_compatible(self, user1: User, user2: User) -> bool:
        age_compatible = user1.preferred_age == self.transform_age(user2.age)
        gender_compatible = (user1.preferred_gender == self.transform_gender(user2.gender)
                           or user1.preferred_gender == 'ANY')
        return age_compatible and gender_compatible

    def calculate_match_score(self, user1: User, user2: User, distance: float) -> float:
        if not self.is_compatible(user1, user2):
            return 0

        pref1 = {k: getattr(user1, k) for k in ['cultural_tourism', 'food', 'historical_tourism',
                                               'leisure_sports', 'nature_tourism', 'shopping']}
        pref2 = {k: getattr(user2, k) for k in ['cultural_tourism', 'food', 'historical_tourism',
                                               'leisure_sports', 'nature_tourism', 'shopping']}

        preference_similarity = self.calculate_preference_similarity(pref1, pref2)
        distance_score = 1 / (1 + distance)

        preference_weight = 0.7
        distance_weight = 0.3

        return (preference_similarity * preference_weight) + (distance_score * distance_weight)

    def generate_match_description(self, user1: User, user2: User, score: float, distance: float) -> str:
        try:
            prompt = f"""
            두 사용자의 매칭 결과에 대한 설명을 생성해주세요. 아래 예시를 참고하여 100자 이내로 작성해주세요.

            예시 1:
            사용자1: 문화관광 4, 음식 5, 역사관광 3, 레저스포츠 2, 자연관광 4, 쇼핑 3
            사용자2: 문화관광 5, 음식 4, 역사관광 4, 레저스포츠 1, 자연관광 5, 쇼핑 2
            매칭 점수: 0.85, 거리: 2.5km
            설명: 두 분은 문화와 자연 관광에 대한 높은 관심을 공유하고 있어요. 근처 박물관 탐방 후 야외 피크닉을 즐겨보는 건 어떨까요?

            예시 2:
            사용자1: 문화관광 2, 음식 5, 역사관광 1, 레저스포츠 4, 자연관광 3, 쇼핑 5
            사용자2: 문화관광 1, 음식 4, 역사관광 2, 레저스포츠 5, 자연관광 3, 쇼핑 4
            매칭 점수: 0.78, 거리: 1.8km
            설명: 음식과 레저스포츠, 쇼핑을 좋아하시는 두 분! 현지 맛집 탐방 후 근처 스포츠 시설에서 함께 운동하는 것은 어떨까요?

            실제 사용자 정보:
            사용자1의 선호도:
            - 문화관광: {user1.cultural_tourism}
            - 음식: {user1.food}
            - 역사관광: {user1.historical_tourism}
            - 레저스포츠: {user1.leisure_sports}
            - 자연관광: {user1.nature_tourism}
            - 쇼핑: {user1.shopping}

            사용자2의 선호도:
            - 문화관광: {user2.cultural_tourism}
            - 음식: {user2.food}
            - 역사관광: {user2.historical_tourism}
            - 레저스포츠: {user2.leisure_sports}
            - 자연관광: {user2.nature_tourism}
            - 쇼핑: {user2.shopping}

            매칭 점수: {score:.2f}
            거리: {distance:.2f}km

            위 정보를 바탕으로 두 사용자가 잘 맞는 이유와 함께 추천하는 활동을 설명해주세요.
            """

            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.7
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating match description: {str(e)}")
            return "두 사용자의 여행 성향이 잘 맞아 즐거운 여행이 될 것 같습니다."

    def get_candidates(self, user_id: str) -> List[User]:
        def db_operation(user_id):
            query = """
            SELECT U.user_id, age, gender, cultural_tourism, food, historical_tourism,
                   leisure_sports, nature_tourism, shopping, preferred_age, preferred_gender,
                   latitude, longitude
            FROM user U JOIN survey S ON U.user_id = S.user_id
            WHERE U.user_id != %s
            """
            with self.conn.cursor(DictCursor) as cursor:
                cursor.execute(query, (user_id,))
                return cursor.fetchall()

        users_data = self.execute_with_retry(db_operation, user_id)
        return [User(user_data) for user_data in users_data]

    def find_best_match(self, user: User, max_distance: float) -> Tuple[User, float, float]:
        candidates = self.get_candidates(user.user_id)

        def process_candidate(candidate):
            if not user.latitude or not user.longitude or not candidate.latitude or not candidate.longitude:
                return None, 0, float('inf')

            distance = self.calculate_distance(user.latitude, user.longitude,
                                            candidate.latitude, candidate.longitude)
            if distance > max_distance:
                return None, 0, float('inf')

            score = self.calculate_match_score(user, candidate, distance)
            return candidate, score, distance

        futures = [executor.submit(process_candidate, candidate) for candidate in candidates]
        best_match, best_score, best_distance = None, 0, float('inf')

        for future in as_completed(futures):
            match, score, distance = future.result()
            if score > best_score:
                best_match, best_score, best_distance = match, score, distance

        return best_match, best_score, best_distance

    def clear_user_coordinates(self, user_id: str):
        def db_operation(user_id):
            query = "UPDATE user SET latitude = NULL, longitude = NULL WHERE user_id = %s"
            with self.conn.cursor() as cursor:
                cursor.execute(query, (user_id,))

        try:
            self.execute_with_retry(db_operation, user_id)
            logger.info(f"Cleared coordinates for user {user_id}")
        except Exception as e:
            logger.error(f"Failed to clear coordinates for user {user_id}: {e}")

    def update_user_coordinates(self, user_id: str, latitude: float, longitude: float):
        def db_operation(user_id, latitude, longitude):
            query = "UPDATE user SET latitude = %s, longitude = %s WHERE user_id = %s"
            with self.conn.cursor() as cursor:
                cursor.execute(query, (latitude, longitude, user_id))

        self.execute_with_retry(db_operation, user_id, latitude, longitude)

    def get_user(self, user_id: str) -> User:
        def db_operation(user_id):
            query = """
            SELECT U.user_id, age, gender, cultural_tourism, food, historical_tourism,
                   leisure_sports, nature_tourism, shopping, preferred_age, preferred_gender,
                   latitude, longitude
            FROM user U JOIN survey S ON U.user_id = S.user_id
            WHERE U.user_id = %s
            """
            with self.conn.cursor(DictCursor) as cursor:
                cursor.execute(query, (user_id,))
                user_data = cursor.fetchone()

            if not user_data:
                raise ValueError(f"User with id {user_id} not found")

            user_data['age'] = self.transform_age(user_data['age'])
            user_data['preferred_age'] = user_data['preferred_age']
            user_data['gender'] = self.transform_gender(user_data['gender'])
            user_data['preferred_gender'] = user_data['preferred_gender']
            return user_data

        return User(self.execute_with_retry(db_operation, user_id))

    def store_match_in_redis(self, user_id: str, match_result: Dict):
        """
        매칭 결과를 Redis에 저장하는 메소드
        """
        try:
            # 매칭 결과를 Redis에 저장 (user_id를 키로 사용)
            self.redis_client.set(f"match:{user_id}", str(match_result))
            logger.info(f"Match result stored in Redis for user {user_id}")
        except Exception as e:
            logger.error(f"Failed to store match result in Redis for user {user_id}: {e}")

    def handle_matching_request(self, request: Dict) -> Dict:
        try:
            user_id = request["user_id"]
            latitude = float(request["latitude"])
            longitude = float(request["longitude"])
            retry = int(request.get("retry", 0))

            logger.debug(
                f"Received request: user_id={user_id}, latitude={latitude}, longitude={longitude}, retry={retry}")

            if user_id in self.matching_results:
                result = self.matching_results.pop(user_id)
                self.clear_user_coordinates(user_id)
                return result

            try:
                user = self.get_user(user_id)
                user.latitude = latitude
                user.longitude = longitude
                self.update_user_coordinates(user_id, latitude, longitude)
            except Exception as e:
                logger.error(f"Error getting/updating user data: {e}")
                return {"result": "error", "message": "Failed to process user data"}

            start_time = time.time()
            best_match, best_score, best_distance = None, 0, float('inf')

            distance_map = {0: 1, 1: 3, 2: 5}
            max_distance = distance_map.get(retry, 1)

            while time.time() - start_time < 10:
                time.sleep(0.5)
                if user_id in self.matching_results:
                    result = self.matching_results.pop(user_id)
                    self.clear_user_coordinates(user_id)
                    return result

                try:
                    current_match, current_score, current_distance = self.find_best_match(user,
                                                                                          max_distance=max_distance)
                    if current_score > best_score:
                        best_match, best_score, best_distance = current_match, current_score, current_distance

                    description = self.generate_match_description(user, best_match, best_score, best_distance)

                    match_result = {
                        "result": "success",
                        "matched_user": best_match.user_id,
                        "score": best_score,
                        "distance": best_distance,
                        "description": description
                    }

                    # 매칭 결과를 Redis에 저장
                    self.store_match_in_redis(user_id, match_result)
                    self.clear_user_coordinates(user_id)
                    return match_result

                except Exception as e:
                    logger.error(f"Error during matching process: {e}")
                    continue



            if user_id in self.matching_results:
                result = self.matching_results.pop(user_id)
                self.clear_user_coordinates(user_id)
                return result

            self.clear_user_coordinates(user_id)
            return {
                "result": "fail",
                "message": "No match found within the time limit"
            }

        except Exception as e:
            logger.error(f"Error handling matching request: {str(e)}")
            try:
                self.clear_user_coordinates(user_id)
            except Exception as clear_error:
                logger.error(f"Failed to clear coordinates after error: {clear_error}")
            return {"result": "error", "message": str(e)}

    def __del__(self):
        """소멸자에서 데이터베이스 연결을 정리"""
        try:
            if self.conn and self.conn.open:
                self.conn.close()
                logger.info("Database connection closed successfully")
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")

matching_service = MatchingService()

@app.route('/match', methods=['POST'])
@swag_from('match.yml')
def match_users():
    try:
        return jsonify(matching_service.handle_matching_request(request.json))
    except Exception as e:
        logger.error(f"Error in match_users endpoint: {e}")
        return jsonify({"result": "error", "message": "Internal server error"}), 500


if __name__ == "__main__":
    try:
        app.run(host='0.0.0.0', port=6000, debug=False)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")