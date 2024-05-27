import telebot
import json
import torch
from modelsp import NeuralNet
from nltk_utils import bag_of_words, tokenize
import random
import logging
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

keyboard1 = telebot.types.ReplyKeyboardMarkup()


bot = telebot.TeleBot('6877858179:AAFWwGypEQ3i-WNRGV-ghNAUOUq7-H_9Z_w')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Переменные для метрик
total_requests = 0
correct_predictions = 0
response_times = []
intent_predictions = []
intent_true_labels = []


@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, 'Привет это FAQ-бот одного из лучших институтов УрФУ! Постараюсь ответить на твой вопрос.', reply_markup=keyboard1)


@bot.message_handler(commands=['metrics'])
def send_metrics(message):
    metrics = calculate_metrics()
    bot.send_message(message.chat.id, metrics)


@bot.message_handler(content_types=['text'])
def send_text(message):
    global total_requests, correct_predictions, response_times
    start_time = time.time()

    sentence = message.text
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    total_requests += 1

    if prob.item() > 0.8:
        intent_predictions.append(tag)
        for intent in intents['intents']:
            if tag == intent["tag"]:
                bot.send_message(message.chat.id, f"{random.choice(intent['responses'])}")
        correct_predictions += 1
        intent_true_labels.append(tag)
    else:
        bot.send_message(message.chat.id, f"Простите, такого я ещё не знаю." + "\n")
        logging.info(f"{bot_name}: I do not understand...")

    response_time = time.time() - start_time
    response_times.append(response_time)

def calculate_metrics():
    if not intent_true_labels or not intent_predictions:
        return "Not enough data to calculate metrics."

    accuracy = accuracy_score(intent_true_labels, intent_predictions)
    precision = precision_score(intent_true_labels, intent_predictions, average='weighted')
    recall = recall_score(intent_true_labels, intent_predictions, average='weighted')
    f1 = f1_score(intent_true_labels, intent_predictions, average='weighted')
    avg_response_time = np.mean(response_times)

    metrics = (
        f"Total Requests: {total_requests}\n"
        f"Correct Predictions: {correct_predictions}\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"F1 Score: {f1:.4f}\n"
        f"Average Response Time: {avg_response_time:.4f} seconds"
    )

    logging.info(metrics)
    return metrics


bot.polling(none_stop=True)
