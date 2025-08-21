# ИМПОРТ БИБЛИОТЕК
import telebot
from random import randint
from datetime import datetime
import time
import random
import telebot
import requests
import os
import gdown
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
from flask import Flask, request

TOKEN = os.getenv("BOT_TOKEN")
bot = telebot.TeleBot(TOKEN, parse_mode=None)

app = Flask(__name__)

@app.route('/')
def index():
    return "Bot is running!"

@app.route(f'/{TOKEN}', methods=['POST'])
def webhook():
    json_str = request.get_data().decode('UTF-8')
    update = telebot.types.Update.de_json(json_str)
    bot.process_new_updates([update])
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
        print("Ошибка сохранения истории: ", e)
        
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
            "Authorization": f"Bearer {os.getenv('API_KEY')}"
        }
        data = {
            "model": "deepseek-ai/DeepSeek-R1-0528",
            "messages": history[str(user_id)]
        }

        response = requests.post(url, headers=headers, json=data)
        data = response.json()

        if 'choices' in data and data['choices']:
            content = data['choices'][0]['message']['content']
            history[str(user_id)].append({"role": "assistant", "content": content})

            if len(history[str(user_id)]) > 16:
                history[str(user_id)] = [history[str(user_id)][0]] + history[str(user_id)][-15:]

            save_history()

            if '</think>' in content:
                return content.split('</think>', 1)[1]
            return content
        else:
            return f"Ошибка API: {data}"
    except Exception as e:
        return f"Ошибка при запросе: {e}"

def cat_dog(photo):
    try:
        np.set_printoptions(suppress=True)
        model_path = "cat_dog_model.h5"
        if not os.path.exists(model_path):
            url = os.getenv("CAT_DOGS_MODEL_URL")
            gdown.download(url, model_path, quiet=False)
        model = load_model(model_path, compile=False)
        image = Image.open(photo).convert("RGB")
        size = (150, 150)
        image = ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)
        image_array = np.asarray(image).astype(np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        prediction = model.predict(image_array)
        confidence = float(prediction[0])
        if confidence < 0.5:
            return f"На изображении кот (точность: {confidence:.2f})"
        else:
            return f"На изображении собака (точность: {confidence:.2f})"
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


def number_identification(photo):
    np.set_printoptions(suppress=True)
    model = load_model("mnist_model.h5", compile=False)
    image = Image.open(photo).convert("L")
    image = ImageOps.invert(image)
    size = (28, 28)
    image = ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)
    image_array = np.asarray(image).astype(np.float32) / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)
    prediction = model.predict(image_array)
    index = np.argmax(prediction)

    return str(index)


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

@bot.message_handler(content_types=['text'])
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
            bot.send_message(message.chat.id, "Думаю над ответом...")
            answer = chat(message.chat.id, message.text)
            bot.send_message(message.chat.id, answer, parse_mode='Markdown')
            bot.delete_message(message.chat.id, message.id+1)
    except Exception as e:
        bot.send_message(message.chat.id, f"Ошибка: {e}")


# ВЫПОЛНЯЕТСЯ, КОГДА ПОЛЬЗОВАТЕЛЬ НАЖИМАЕТ НА ОДНУ ИЗ КНОПОК КЛАВИАТУРЫ InlineKeyboard
@bot.callback_query_handler(func=lambda call: call.data in ('1', '2', '3', '4', '5', '6'))
def answer(call):
    value = bot.send_dice(call.message.chat.id, emoji='').dice.value
    if str(value) == call.data:
        bot.send_message(call.message.chat.id, "Победа!")
    else:
        bot.send_message(call.message.chat.id, "Попробуй еще раз")



if __name__ == "__main__":
    server_url = os.getenv("RENDER_EXTERNAL_URL")
    if server_url and TOKEN:
        webhook_url = f"{server_url}/{TOKEN}"
        set_webhook_url = f"https://api.telegram.org/bot{TOKEN}/setWebhook?url={webhook_url}"
        try:
            r = requests.get(set_webhook_url)
            print("Webhook установлен:", r.text)
        except Exception as e:
            print("Ошибка при установке webhook:", e)

        port = int(os.environ.get("PORT", 10000))
        print(f"Starting server on port {port}")
        app.run(host='0.0.0.0', port=port)
    else:
        print("Запуск бота в режиме pooling")
        bot.remove_webhook()
        bot.polling(none_stop=True)
