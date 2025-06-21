## ВАРИАНТ кода для генерации новостей с обновлённой библиотекой OpenAI

import os
from typing import Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from dotenv import load_dotenv
from openai import OpenAI

# Загружаем переменные окружения из .env файла для локальной разработки
load_dotenv()

app = FastAPI()

# Получаем API-ключи из переменных окружения
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CURRENTS_API_KEY = os.getenv("CURRENTS_API_KEY")

# Проверяем наличие ключей, иначе останавливаем приложение
if not OPENAI_API_KEY or not CURRENTS_API_KEY:
    raise RuntimeError(
        "Необходимо указать переменные окружения OPENAI_API_KEY и CURRENTS_API_KEY"
    )

# Инициализация клиента OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

class Topic(BaseModel):
    """
    Модель запроса для генерации поста.
    """
    topic: str

def get_recent_news(topic: str) -> str:
    """
    Получает свежие новости по теме через Currents API.
    Возвращает заголовки 5 новостей или сообщение об отсутствии новостей.
    """
    url = "https://api.currentsapi.services/v1/latest-news"
    params = {
        "language": "en",
        "keywords": topic,
        "apiKey": CURRENTS_API_KEY
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        news_data = response.json().get("news", [])
        if not news_data:
            return "Свежих новостей не найдено."
        return "\n".join([article["title"] for article in news_data[:5]])
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Ошибка запроса к Currents API: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке новостей: {str(e)}")

def generate_content(topic: str) -> Dict[str, str]:
    """
    Генерирует заголовок, мета-описание и статью по теме с помощью OpenAI.
    Использует последние новости как контекст.
    """
    recent_news = get_recent_news(topic)

    try:
        # Генерируем заголовок
        title_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"Придумайте привлекательный и точный заголовок для статьи на тему '{topic}', с учётом актуальных новостей:\n{recent_news}."
            }],
            max_tokens=60,
            temperature=0.5,
            stop=["\n"]
        )
        title = title_response.choices[0].message.content.strip()

        # Генерируем мета-описание
        meta_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"Напишите мета-описание для статьи с заголовком: '{title}'. Оно должно быть полным, информативным и содержать основные ключевые слова."
            }],
            max_tokens=120,
            temperature=0.5,
            stop=["."]
        )
        meta_description = meta_response.choices[0].message.content.strip()

        # Генерируем содержимое статьи
        post_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": (
                    f"Напишите подробную статью на тему '{topic}', используя последние новости:\n{recent_news}. "
                    "Статья должна быть:\n"
                    "1. Информативной и логичной\n"
                    "2. Не менее 1500 символов\n"
                    "3. С четкой структурой с подзаголовками\n"
                    "4. С анализом текущих трендов\n"
                    "5. Со вступлением, основной частью и заключением\n"
                    "6. С примерами из актуальных новостей\n"
                    "7. Каждый абзац — не менее 3-4 предложений\n"
                    "8. Текст — легким для восприятия"
                )
            }],
            max_tokens=1500,
            temperature=0.5,
            presence_penalty=0.6,
            frequency_penalty=0.6
        )
        post_content = post_response.choices[0].message.content.strip()

        return {
            "title": title,
            "meta_description": meta_description,
            "post_content": post_content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации текста: {str(e)}")

@app.post("/generate-post")
async def generate_post_api(topic: Topic):
    """
    Эндпоинт для генерации блог-поста по теме.
    """
    return generate_content(topic.topic)

@app.get("/")
async def root():
    """
    Корневой эндпоинт для проверки работоспособности сервиса.
    """
    return {"message": "Сервис работает"}

@app.get("/heartbeat")
async def heartbeat_api():
    """
    Эндпоинт для проверки состояния сервиса.
    """
    return {"status": "OK"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port)

