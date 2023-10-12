import pickle
# import numpy as np
from flask import Flask, request

import sys
from asizeof import asizeof as better_size

# vocab = None  # vocab (TfidfVectorizer) is too big (~0.5GB), so let's try to work without it
model = None

vocab_dict : dict[str,int] = None
vocab_list : list[str] = None

app = Flask(__name__)

header = """
<h1> <span style="background-color:rgb(255, 255, 255)">
Movie Review Sentiment by Matthew Inglis-Whalen
</span> </h1>
<p>
<span style="background-color:#00FF00"> Good </span>
<span style="background-color:#AAFFAA"> Okay </span> 
<span style="background-color:#FFFFFF"> Neutral </span> 
<span style="background-color:#FF7777"> Bad </span> 
<span style="background-color:#FF0000"> Terrible </span> 
</p>
"""
last_return = ""

empty_star = "✰"
filled_star = "★"

def prob_to_html(r: float) -> str :
    """Presents the model's predicted probability as html markup"""
    return_str = "<p> Predicted stars: "
    for n in range(10) :
        if n/10 + 0.05 < r :
            return_str += filled_star
        else :
            return_str += empty_star

    return_str += "</p> <p>Sentiment: "
    if r < 0.25 :
        return_str += "Bad"
    elif r > 0.75 :
        return_str += "Good"
    else :
        return_str += "Neutral"
    return_str += "</p>"
    return return_str

def strength_to_color_string(strength: float, min_str: float, max_str: float) -> str :
    """Returns a color choice as a string in the format #XXXXXX.
        Very green for high positive strength, very red for high negative strength"""
    strength = max(min_str, min(max_str, strength))  # clamp strength to min < strength < max
    color_str = "#"
    if strength < 0 :
        color_str += "FF"                                               # red
        color_str += "%0.2X" % int( (min_str-strength)/min_str * 255)   # green
        color_str += "%0.2X" % int( (min_str-strength)/min_str * 255)   # blue
    else:
        color_str += "%0.2X" % int( (max_str-strength)/max_str * 255)   # red
        color_str += "FF"                                               # green
        color_str += "%0.2X" % int( (max_str-strength)/max_str * 255)   # blue
    return color_str

def transformed_word_to_list(transformed_word) :
    """Turns a sparse array into a list of strings"""
    my_list = [(vocab_list[j], transformed_word[i, j]) for i, j in zip(*transformed_word.nonzero())]
    return my_list

def strength_1gram(bword) :
    """Turn a binary word into a strength based on logistic regression coefficient of that word"""
    word = bword.decode()  # bit string to string
    word_as_sparse_arr = vocab.transform([word])
    _, word_index_as_list = word_as_sparse_arr.nonzero()
    if word_index_as_list.size > 0:
        strength = model.coef_[0, word_index_as_list[0]]
    else :
        strength = 0
    return strength

def strength_2gram(bword1,bword2) :
    """Turns two binary words into a strength based on logistic regression coefficient of that 2gram"""
    word1, word2 = bword1.decode(), bword2.decode()  # bit string to string
    bigram_as_sparse_arr = vocab.transform([f"{word1} {word2}"])
    _, bigram_index_as_list = bigram_as_sparse_arr.nonzero()
    if bigram_index_as_list.size > 0 :
        strength = model.coef_[0, bigram_index_as_list[0]]
    else :
        strength = 0
    return strength


def reasoning_html_from_string(data: str) -> str :
    """Returns html markup containing the data string with coloring based on model coefficients"""
    reasoning_str = "<p> Reasoning: "
    min_str, max_str = min(model.coef_[0]), max(model.coef_[0])

    # pad the data with meaningless buffer strings
    data_list = [b"BBBB"]
    data_list.extend(data.split())
    data_list.extend([b"BBBB"])

    for n, (prev_bword, this_bword, next_bword) in enumerate( zip(data_list[:-2],data_list[1:-1],data_list[2:])):

        this_word = this_bword.decode()
        str_1gram = strength_1gram(this_bword)
        str_2gram_prev = strength_2gram(prev_bword,this_bword)
        str_2gram_next = strength_2gram(this_bword,next_bword)

        net_strength = (str_1gram+str_2gram_prev+str_2gram_next)/3
        right_strength = (str_1gram+str_2gram_next)/3
        reasoning_str += f"""<span style = "background-color:{strength_to_color_string(net_strength,min_str,max_str)}">{this_word}</span>"""
        reasoning_str += f"""<span style = "background-color:{strength_to_color_string(right_strength,min_str,max_str)}"> </span>\n"""
        if n % 12 == 11 :
            reasoning_str += "<br>"
    reasoning_str += "</p>"
    return reasoning_str

def load_model():
    """Load the stored inference model and vocabulary"""
    global vocab
    global model
    global vocab_dict
    global vocab_list
    with open('vocabulary.pkl', 'rb') as f_v:
        vocab = pickle.load(f_v)
    with open('sentiment_inference_model.pkl', 'rb') as f_m:
        model = pickle.load(f_m)
    vocab_dict = vocab.vocabulary_
    vocab_list = vocab.get_feature_names_out()

@app.route('/')
def home_endpoint():
    """Locally: what to show when visiting localhost:80"""
    return header+last_return


@app.route('/predict', methods=['POST'])
def get_prediction():
    """Locally: what to show when receiving a post request at localhost:80/predict"""
    """>  curl.exe -X POST localhost:80/predict -H 'Content-Type: application/json' -d 'This is a review' """
    global last_return
    # Works only for a single sample
    if request.method == 'POST':
        data = request.get_data()  # Get data posted as a string
        transformed_data = vocab.transform([data])  # Transform the input string with the prefit TfIdfVectorizer
        sentiment = model.predict_proba(transformed_data)[0,1]  # Runs globally-loaded model on the data
        last_return = prob_to_html(sentiment) + reasoning_html_from_string(data)

    return last_return


if __name__ == '__main__':
    load_model()  # load model at the beginning once only

    print(f"model: {sys.getsizeof(model)} {better_size(model) / 1e3:.1f}KB")
    print(f"vocab: {sys.getsizeof(vocab)} {better_size(vocab) / 1e6:.1f}MB")
    print(f"vocab dict: {sys.getsizeof(vocab_dict)} {better_size(vocab_dict)/ 1e6:.1f}MB")
    print(f"vocab list: {sys.getsizeof(vocab_list)} {better_size(vocab_list)/ 1e6:.1f}MB")

    app.run(host='0.0.0.0', port=80)

