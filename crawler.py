import requests
from sympy.integrals.meijerint_doc import category

"""
from bs4 import BeautifulSoup

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36'}

# 이 이후로 필요한 부분 추출
data = requests.get('https://korean.visitseoul.net/attractions?srchType=&srchOptnCode=&srchCtgry=68&sortOrder=&srchWord=&radioOptionLike=TURSM_AREA_8',headers=headers)
soup = BeautifulSoup(data.text, 'html.parser')

print(soup.prettify())

title = soup.select_one('#li.item:nth-child(1) > a:nth-child(1) > div:nth-child(2) > div:nth-child(1) > span:nth-child(1)')
print(title)
"""

from bs4 import BeautifulSoup

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36'
}

category = ["카페&디저트", "주점", "한식", "양식", "중식", "일식", "아시아식", "서양식", "채식", "할랄"]

# BeautifulSoup 객체 생성
for i in range(1, 60):
    try:
        html = requests.get(f'https://korean.visitseoul.net/area?curPage={i}&srchType=&srchOptnCode=&srchCtgry=98&sortOrder=&srchWord=&radioOptionLike=TURSM_AREA_8',headers=headers)
    except requests.RequestException as e:
        print(f'Error: {e}')
        break

    soup = BeautifulSoup(html.text, 'html.parser')

    # 모든 `li` 요소 찾기
    items = soup.find_all('li', class_='item')

    # 정보 추출
    for item in items:
        title = item.find('span', class_='title')
        description = item.find('span', class_='small-text text-dot-d')

        # 텍스트가 있으면 출력
        if title and description:
            print("Title:", title.get_text(strip=True))
            print("Location: 기타")
            print("Description:", description.get_text(strip=True))
            print('-' * 30)
"""
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36'
}

html = requests.get(f'https://korean.visitseoul.net/attractions?curPage=2&srchType=&srchOptnCode=&srchCtgry=69&sortOrder=&srchWord=&radioOptionLike=TURSM_AREA_8',headers=headers)
soup = BeautifulSoup(html.text, 'html.parser')

# 모든 `li` 요소 찾기
items = soup.find_all('li', class_='item')

# 정보 추출
for item in items:
    title = item.find('span', class_='title')
    description = item.find('span', class_='small-text text-dot-d')

    # 텍스트가 있으면 출력
    if title and description:
        print("Title:", title.get_text(strip=True))
        print("Description:", description.get_text(strip=True))
        print('-' * 30)
"""