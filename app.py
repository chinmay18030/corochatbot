import tf.keras as keras
import nltk
import numpy as np
from flask import Flask, render_template, request
from static.Data.data import words, classes, answers
from nltk.stem import WordNetLemmatizer

model = keras.models.load_model('static/Models/Covibot.model')
lemmatizer = WordNetLemmatizer()


def clean_sentence(sentence):
    sentence_word = nltk.word_tokenize(sentence)
    sentence_word = [lemmatizer.lemmatize(word) for word in sentence_word]
    return sentence_word


def bag_of_words(sentence):
    sw = clean_sentence(sentence)
    bag = [0] * len(words)
    for w in sw:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(model, sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    results = [[i, r] for i, r in enumerate(res) if r > 0.25]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def get_response(intent_list):
    tag = intent_list[0]["intent"]
    return answers[tag]



app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def weather():
    if request.method == 'POST':
        city = request.form['city']
    else:
        # for default name mathura
        city = ""


    ints = predict_class(model, city.lower())
    if ints == []:
        data = {
            "Probability":"",
            "answer":""
        }
    else:
        res = get_response(ints)
        data = {
            "City":city,
           "Probability": str(int(float(ints[0]["probability"])*100.0))+"%",
            "answer": res
        }
    print(ints)



    return render_template('index.html',data=data)


if __name__ == '__main__':
    app.run(debug=True)
