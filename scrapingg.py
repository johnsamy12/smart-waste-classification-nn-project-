import os
import json
import time
import random
import requests
import pandas as pd
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO

OUTPUT_DIR = "final_data"
MIN_SIZE = (100, 100)
TARGET_PER_CLASS = 500  

CATEGORIES = {
    "glass": ["glass waste", "glass bottle trash", "broken glass item for recycling", "glass jar or bottle thrown away", "shattered glass piece in garbage bin"],
    "paper": ["paper waste", "newspaper trash", "crumpled paper waste", "white paper material discarded", "torn paper piece in waste bin"],
    "metal": ["metal can waste", "aluminum trash", "crushed metal can in trash", "used tin can discarded", "steel or aluminum waste object"],
    "plastic": ["plastic bottle in garbage bin", "plastic waste item for recycling", "used plastic container", "plastic packaging trash", "plastic bag found in garbage"],
}

HEADERS = {"User-Agent": "Mozilla/5.0"}

def get_urls(query, limit=100):
    urls = []
    offset = 0
    # Bing usually provides ~35 images per page
    while len(urls) < limit:
        try:
            r = requests.get(
                "https://www.bing.com/images/search",
                params={"q": query, "first": offset},
                headers=HEADERS,
                timeout=10
            )
            soup = BeautifulSoup(r.text, "html.parser")
            items = soup.find_all("a", class_="iusc")

            if not items:
                break

            for item in items:
                if len(urls) >= limit:
                    break
                data = json.loads(item.get("m"))
                url = data.get("murl")
                if url and url.startswith("http"):
                    urls.append(url)
            
            offset += 35
            time.sleep(0.5) 
        except Exception:
            break
    return urls

def save_img(url, path):
    try:
        r = requests.get(url, timeout=5)
        img = Image.open(BytesIO(r.content))
        img.verify() # Check if it's a valid image

        img = Image.open(BytesIO(r.content)) # Re-open after verify
        if img.size[0] < MIN_SIZE[0] or img.size[1] < MIN_SIZE[1]:
            return False

        img.convert("RGB").save(path, "JPEG")
        return True
    except:
        return False

def scrape():
    for cls, queries in CATEGORIES.items():
        folder = os.path.join(OUTPUT_DIR, cls)
        os.makedirs(folder, exist_ok=True)

        
        current_count = len([f for f in os.listdir(folder) if f.endswith('.jpg')])
        print(f"Starting category: {cls}. Current count: {current_count}")

        for q in queries:
            if current_count >= TARGET_PER_CLASS:
                break
            
            
            needed = TARGET_PER_CLASS - current_count
            #
            urls = get_urls(q, limit=needed + 50) 

            for url in urls:
                if current_count >= TARGET_PER_CLASS:
                    break
                
                name = f"{current_count}.jpg"
                path = os.path.join(folder, name)

                if save_img(url, path):
                    current_count += 1
                    if current_count % 10 == 0:
                        print(f"  {cls}: {current_count}/{TARGET_PER_CLASS}")

                time.sleep(random.uniform(0.1, 0.3))

def clean():
    for cls in os.listdir(OUTPUT_DIR):
        folder = os.path.join(OUTPUT_DIR, cls)

        for f in os.listdir(folder):
            path = os.path.join(folder, f)
            try:
                with Image.open(path) as img:
                    img.verify()
            except:
                os.remove(path)

def build_csv():
    data = []

    for cls in os.listdir(OUTPUT_DIR):
        folder = os.path.join(OUTPUT_DIR, cls)

        for f in os.listdir(folder):
            path = os.path.join(folder, f)
            data.append([path, cls])

    df = pd.DataFrame(data, columns=["image", "label"])
    df.to_csv("dataset2.csv", index=False)

if __name__ == "__main__":
    scrape()
    clean()
    build_csv()