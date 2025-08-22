import os
import sys
import re
import json
import logging
import requests
import gdown
import numpy as np
from random import randint
from datetime import datetime
from flask import Flask, request
from PIL import Image, ImageOps
import telebot
from tensorflow.keras.models import load_model
import tensorflow as tf
from telebot import util

logging.basicConfig(level=logging.INFO)

TOKEN = os.getenv("BOT_TOKEN")
if not TOKEN:
    sys.exit("Ошибка: BOT_TOKEN не задан в переменных окружения")

bot = telebot.TeleBot(TOKEN, parse_mode=None)
app = Flask(__name__)

MAX_LEN = 4096

def convert_markdown_to_html(text: str) -> str:
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)   
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)       
    text = re.sub(r'__(.*?)__', r'<u>\1</u>', text)       
    text = re.sub(r'~~(.*?)~~', r'<s>\1</s>', text)       
    text = re.sub(r'`([^`]*)`', r'<code>\1</code>', text) 
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', text)
    return text
    
def send_long_message(chat_id, text, parse_mode='HTML'):
    try:
        safe_text = convert_markdown_to_html(text or "")
        for part in util.smart_split(safe_text, MAX_LEN):
            bot.send_message(chat_id, part, parse_mode=parse_mode)
    except Exception as e:
        logging.error(f"Ошибка: {e}")

@app.route('/')
def index():
    return "Bot is running!"

@app.route(f'/{TOKEN}', methods=['POST'])
def webhook():
    try:
        json_str = request.get_data(as_text=True)
        update = telebot.types.Update.de_json(json_str)
        if update:
            bot.process_new_updates([update])
    except Exception as e:
        app.logger.exception("Webhook error: %s", e)
    return '', 200

def load_photo(message, name):
    photo = message.photo[-1]
    file_info = bot.get_file(photo.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    save_path = name
    with open(save_path, 'wb') as new_file:
        new_file.write(downloaded_file)

history_file = "history.json"
history = {}

if os.path.exists(history_file):
    try:
        with open(history_file, "r", encoding='utf-8') as f:
            history = json.load(f)
    except Exception:
        history = {}

def save_history():
    try:
        with open(history_file, "w", encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error("Ошибка сохранения истории: %s", e)

API_KEY = os.getenv('API_KEY')
if not API_KEY:
    logging.warning("API_KEY не задан: чат-модель будет недоступна")
    
def chat(user_id, text):
    try:
        if str(user_id) not in history:
            history[str(user_id)] = [
                {"role": "system", "content": "Ты — дружелюбный помощник."}
            ]

        history[str(user_id)].append({"role": "user", "content": text})

        if len(history[str(user_id)]) > 16:
            history[str(user_id)] = [history[str(user_id)][0]] + history[str(user_id)][-15:]

        url = "https://api.intelligence.io.solutions/api/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}" if API_KEY else ""
        }
        data = {
            "model": "deepseek-ai/DeepSeek-R1-0528",
            "messages": history[str(user_id)]
        }

        response = requests.post(url, headers=headers, json=data, timeout=300)
        data = response.json()

        if isinstance(data, dict) and data.get('choices'):
            content = data['choices'][0]['message']['content']
            history[str(user_id)].append({"role": "assistant", "content": content})

            if len(history[str(user_id)]) > 16:
                history[str(user_id)] = [history[str(user_id)][0]] + history[str(user_id)][-15:]

            save_history()

            if '</think>' in content:
                return content.split('</think>', 1)[1]
            return content
        else:
            logging.error(f"Ошибка API: {json.dumps(data, ensure_ascii=False)}")
            bot.send_long_message(f"Ошибка при запросе: {e}, повторите попытку позже")
    except Exception as e:
        logging.error(f"Ошибка при запросе: {e}")
        bot.send_long_message(f"Ошибка при запросе: {e}, повторите попытку позже")

TFLITE_PATH = "cat_dog_model.tflite"
TFLITE_URL = os.getenv("CAT_DOGS_TFLITE_URL")
_interpreter = None
_input_details = None
_output_details = None

def ensure_catdog_tflite():
    global _interpreter, _input_details, _output_details
    if _interpreter is None:
        if not os.path.exists(TFLITE_PATH):
            if not TFLITE_URL:
                raise RuntimeError("CAT_DOGS_TFLITE_URL не задан, а локальной модели нет")
            gdown.download(TFLITE_URL, TFLITE_PATH, quiet=False)

        _interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
        _interpreter.allocate_tensors()
        _input_details = _interpreter.get_input_details()
        _output_details = _interpreter.get_output_details()
    return _interpreter, _input_details, _output_details
    
def cat_dog(photo):
    try:
        interpreter, input_details, output_details = ensure_catdog_tflite()

        image = Image.open(photo).convert("RGB")
        image = ImageOps.fit(image, (150, 150), method=Image.Resampling.LANCZOS)
        x = (np.asarray(image).astype(np.float32) / 255.0)[None, ...]

        interpreter.set_tensor(input_details[0]['index'], x)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])

        if pred.ndim == 2 and pred.shape[1] == 1:
            confidence = float(pred[0][0])
        elif pred.ndim == 1:
            confidence = float(pred[0])
        else:
            confidence = float(np.ravel(pred)[0])

        return (f"На изображении собака (точность: {confidence:.2f})"
                if confidence >= 0.5 else
                f"На изображении кот (точность: {1-confidence:.2f})")
    except Exception as e:
        return f"Ошибка при распознавании: {e}"

def ident_number(message):
    load_photo(message, "Number.jpg")
    answer_number = number_identification("Number.jpg")
    bot.send_message(message.chat.id, f"Цифра на фото: {answer_number}")

def ident_cat_dog(message):
    load_photo(message, "cat_dog.jpg")
    answer = cat_dog("cat_dog.jpg")
    bot.send_message(message.chat.id, answer)


MNIST_PATH = "mnist_model.h5"
_mnist_model = None

def ensure_mnist():
    global _mnist_model
    if _mnist_model is None:
        if not os.path.exists(MNIST_PATH):
            raise RuntimeError("MNIST модель не найдена: mnist_model.h5")
        _mnist_model = load_model(MNIST_PATH, compile=False)
    return _mnist_model

def number_identification(photo):
    try:
        model = ensure_mnist()
        image = Image.open(photo).convert("L")
        image = ImageOps.invert(image)
        image = ImageOps.fit(image, (28, 28), method=Image.Resampling.LANCZOS)
        x = (np.asarray(image).astype(np.float32) / 255.0).reshape(1, 28, 28, 1)
        pred = model.predict(x, verbose=0)
        return str(int(np.argmax(pred)))
    except Exception as e:
        return f"Ошибка распознавания цифры: {e}"


@bot.message_handler(commands=['start'])
def start(message):
    try:
        keyboard = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
        button1 = telebot.types.KeyboardButton(text="Игра в кубик")
        button2 = telebot.types.KeyboardButton(text="Игровой автомат")
        button6 = telebot.types.KeyboardButton(text="Распознавание цифр")
        button7 = telebot.types.KeyboardButton(text="Распознавание животных")
        keyboard.add(button6, button7)
        bot.send_message(message.chat.id, "Привет, я бот с интегрированной моделью DeepSeek! Задай мне вопрос", reply_markup=keyboard)
    except Exception as e:
        bot.reply_to(message, f"Ошибка: {e}")

@bot.message_handler(commands=['date'])
def date(message):
    try:
        bot.send_message(message.chat.id, "Сейчас: "+str(datetime.today()))
    except Exception as e:
        bot.reply_to(message, f"Ошибка: {e}")

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        temp_path = "temp.jpg"
        with open(temp_path, 'wb') as new_file:
            new_file.write(downloaded_file)
        result = cat_dog(temp_path)
        bot.send_message(message.chat.id, result)
        os.remove(temp_path)
    except Exception as e:
        bot.send_message(message.chat.id, f"Ошибка обработки фото: {e}")

@bot.message_handler(func=lambda message: True, content_types=['text'])
def handle_text(message):
    try:
        text = message.text

        if text == "Игра в кубик":
            keyboard2 = telebot.types.InlineKeyboardMarkup(row_width=3)
            button1 = telebot.types.InlineKeyboardButton("1", callback_data='1')
            button2 = telebot.types.InlineKeyboardButton("2", callback_data='2')
            button3 = telebot.types.InlineKeyboardButton("3", callback_data='3')
            button4 = telebot.types.InlineKeyboardButton("4", callback_data='4')
            button5 = telebot.types.InlineKeyboardButton("5", callback_data='5')
            button6 = telebot.types.InlineKeyboardButton("6", callback_data='6')
            keyboard2.add(button1, button2, button3, button4, button5, button6)
            bot.send_message(message.chat.id, "Угадай число на кубике", reply_markup=keyboard2)
        elif text == "Игровой автомат":
            value = bot.send_dice(message.chat.id, emoji='🎰').dice.value
            if value in (1, 22, 43, 16, 32, 48):
                bot.send_message(message.chat.id, "Победа!")
            elif value == 64:
                bot.send_message(message.chat.id, "Jackpot!")
            else:
                bot.send_message(message.chat.id, "Попробуй еще раз")       
        elif text == "Распознавание цифр":
            send1 = bot.send_message(message.chat.id, "Загрузите изображение цифры")
            bot.register_next_step_handler(send1, ident_number)
        elif text == "Распознавание животных":
            send2 = bot.send_message(message.chat.id, "Загрузите изображение кошки или собаки")
            bot.register_next_step_handler(send2, ident_cat_dog)
        else:
            msg = bot.send_message(message.chat.id, "Думаю над ответом…")
            try:
                answer = chat(message.chat.id, message.text)
                send_long_message(message.chat.id, answer)
            finally:
                try:
                    bot.delete_message(message.chat.id, msg.message_id)
                except Exception:
                    pass
    except Exception as e:
        bot.send_message(message.chat.id, f"Ошибка: {e}")

@bot.callback_query_handler(func=lambda call: call.data in ('1', '2', '3', '4', '5', '6'))
def answer(call):
    value = bot.send_dice(call.message.chat.id, emoji='🎲').dice.value
    if str(value) == call.data:
        bot.send_message(call.message.chat.id, "Победа!")
    else:
        bot.send_message(call.message.chat.id, "Попробуй еще раз")

if __name__ == "__main__":
    server_url = os.getenv("RENDER_EXTERNAL_URL")
    if server_url and TOKEN:
        webhook_url = f"{server_url.rstrip('/')}/{TOKEN}"
        try:
            r = requests.get(f"https://api.telegram.org/bot{TOKEN}/setWebhook",
                             params={"url": webhook_url}, timeout=10)
            logging.info("Webhook установлен: %s", r.text)
        except Exception:
            logging.exception("Ошибка при установке webhook")
        port = int(os.environ.get("PORT", 10000))
        logging.info("Starting server on port %s", port)
        app.run(host='0.0.0.0', port=port)
    else:
        logging.info("Запуск бота в режиме polling")
        bot.remove_webhook()
        bot.infinity_polling(timeout=60)
