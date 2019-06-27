from model import ClassPredictor
import telegram
from telegram_token import token
import torch
import numpy as np
from PIL import Image
from io import BytesIO

model = ClassPredictor()

def start(bot, update):
    #Реализуем приветсвтие бота
    bot.sendMessage(chat_id = update.message.chat_id, 
                    text = "Привет!🌈\nЯ бот, который умеет определять какому художнику принадлежит картина!👨🏻‍🎨\n\nСейчас я умею определять 53 наиболее известных художников всех времён!")
    bot.sendMessage(chat_id = update.message.chat_id, 
                    text = "Вывести список художников можно командой /list")
    bot.sendMessage(chat_id = update.message.chat_id, 
                    text = "Отправь мне фото картины, чтобы узнать, какой художник её написал!")
    bot.sendMessage(chat_id = update.message.chat_id, 
                    text = "Если ты делаешь фотографию картины, обрежь на ней весь фон, кроме самой картины.\n\n<b>Это значительно улучшит определение художника</b>",
                    parse_mode=telegram.ParseMode.HTML)
        
def catalog(bot, update):
    #вывод всех художников по команде /list
    bot.sendMessage(chat_id=update.message.chat_id, 
                    text = "Энди Уорхол\nМарк Шагал\nЛеонардо да Винчи\nКаземир Малевич\nЖоан Миро\nЯн Ван Эйк\nДжескон Поллок\nИероним Босх\nАнри Руссо\nАнри Матисс\nАнри де Тулуз-Лотрек\nГустав Климт\nГюстав Курбе\nДжотто ди Бондоне\nЖорж Сёра\nФрида Кало\nФрансиско Гойя\nЭжен Делакруа\nЭль Греко\nЭдвард Мунк\nЭдуард Мане\nЭдгар Дега\nДиего Веласкес\nДиего Ривера\nКлод Моне\nКараваджо\nКамиль Писсарро\nАндрей Рублёв\nАмедео Модильяни\nАльфред Сислей\nАльбрехт Дюрер\nУильям Тёрнер\nВинсент ван Гог\nВасилий Кандинский\nТициан\nСандро Ботттичелли\nСальвадор Дали\nРене Маргритт\nРембрандт\nРафаэль\nПит Мондриан\nПитер Брейгель\nПьер Огюст Ренуар\nПитер Пауль Рубенс\nПауль Клее\nПоль Гоген\nПабло Пикассо\nПоль Сезанн\nМихаил Врубель\nМикеланджело\nИван Айвазовский\nИван Шишкин\nИлья Репин", 
                    parse_mode=telegram.ParseMode.HTML)

def send_prediction_on_photo(bot, update):
    chat_id = update.message.chat_id
    print("Got image from {}".format(chat_id))

    # получаем информацию о картинке
    image_info = update.message.photo[-1]
    image_file = bot.get_file(image_info)
    image_stream = BytesIO()
    image_file.download(out=image_stream)

    class_ = model.predict(image_stream)
    prob = model.predict_proba(image_stream)
    
    # теперь отправим результат
    update.message.reply_text('\nC вероятностью: ' + str(prob) + '%\nЯ думаю, что это ' + str(class_))
    print("Sent Answer to user, predicted: {}".format(class_))

if __name__ == '__main__':
    from telegram.ext import Updater, MessageHandler, Filters, CommandHandler
    import logging

    # Включим самый базовый логгинг, чтобы видеть сообщения об ошибках
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO)
    
    # используем прокси, так как без него у меня ничего не работало
    REQUEST_KWARGS={
    'proxy_url': 'socks5://173.249.158.25:64621', 
    'urllib3_proxy_kwargs': {
    'username': '***',
    'password': '***',
    }
}
    updater = Updater(token=token, request_kwargs=REQUEST_KWARGS)
    
    #start
    start_handler = CommandHandler('start', start)
    updater.dispatcher.add_handler(start_handler)
    
    #list
    list_handler = CommandHandler('list', catalog)
    updater.dispatcher.add_handler(list_handler)
    
    #обработка фото
    updater.dispatcher.add_handler(MessageHandler(Filters.photo, send_prediction_on_photo))
    updater.start_polling()
