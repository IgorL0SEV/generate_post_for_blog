## УЛУЧШЕННЫЙ ВАРИАНТ кода для генерации новостей

import os
from typing import Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import requests
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла (для локальной разработки)
load_dotenv()

app = FastAPI()

# Получаем ключи API из переменных окружения
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CURRENTS_API_KEY = os.getenv("CURRENTS_API_KEY")

# Проверяем, что ключи существуют, иначе не запускаем сервис
if not OPENAI_API_KEY or not CURRENTS_API_KEY:
    raise RuntimeError(
        "Необходимо указать переменные окружения OPENAI_API_KEY и CURRENTS_API_KEY"
    )

openai.api_key = OPENAI_API_KEY

class Topic(BaseModel):
    """Модель данных для передачи темы в запросе."""
    topic: str

def get_recent_news(topic: str) -> str:
    """
    Получает свежие новости по теме с помощью Currents API.
    Возвращает заголовки 5 новостей (или сообщение об отсутствии).
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
        # Логируем ошибку (можно расширить логирование)
        raise HTTPException(status_code=502, detail=f"Ошибка запроса к Currents API: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке новостей: {str(e)}")

def generate_content(topic: str) -> Dict[str, str]:
    """
    Генерирует заголовок, мета-описание и статью по теме с помощью OpenAI API.
    Использует новости как контекст.
    """
    recent_news = get_recent_news(topic)

    try:
        # Генерируем заголовок
        title_response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
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
        meta_response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"Напишите мета-описание для статьи с заголовком: '{title}'. Оно должно быть полным, информативным и содержать основные ключевые слова."
            }],
            max_tokens=120,
            temperature=0.5,
            stop=["."]
        )
        meta_description = meta_response.choices[0].message.content.strip()

        # Генерируем основное содержимое статьи
        post_response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
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
        # Если произошла ошибка при генерации
        raise HTTPException(status_code=500, detail=f"Ошибка генерации текста: {str(e)}")

@app.post("/generate-post")
async def generate_post_api(topic: Topic):
    """
    Эндпоинт для генерации блог-поста по заданной теме.
    """
    return generate_content(topic.topic)

@app.get("/")
async def root():
    """
    Корневой эндпоинт для проверки статуса сервиса.
    """
    return {"message": "Сервис работает"}

@app.get("/heartbeat")
async def heartbeat_api():
    """
    Эндпоинт для проверки "живости" сервиса.
    """
    return {"status": "OK"}

if __name__ == "__main__":
    import uvicorn
    # Запуск сервера на всех интерфейсах (порт можно задать переменной окружения PORT)
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)

