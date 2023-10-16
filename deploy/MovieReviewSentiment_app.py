import pickle
from flask import Flask, request, make_response, jsonify

vocab = None  # HashingVectorizer
model = None  # LogisticRegression

expected_request_origin = "https://mattingliswhalen.github.io"
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

def strength_gram(*words) :
    """Turns a list of words into a strength based on logistic regression coefficient of that 2gram"""
    bag_of_words = vocab.fit_transform([' '.join(list(words))])

    # the following will give 3 entries: 1 for the 2 1-grams and 1 for the 2-gram
    _, indices = bag_of_words.nonzero()
    net_strength = 0
    for idx in indices :
        bag_val = bag_of_words[0,idx]
        sign = bag_val / abs(bag_val) if abs(bag_val) > 1e-10 else 0
        net_strength += sign * model.coef_[0, idx]

    return net_strength

def reasoning_html_from_string(data: str) -> str :
    """Returns html markup containing the data string with coloring based on model coefficients"""
    reasoning_str = "<p> Reasoning: "
    # min_str, max_str = min(model.coef_[0]), max(model.coef_[0])
    min_str, max_str = -10, 10

    # pad the data with meaningless buffer strings
    data_list = ("BBBB " + data + " BBBB").split()

    for n, (prev_word, this_word, next_word) in enumerate( zip(data_list[:-2],data_list[1:-1],data_list[2:])):

        str_prev = strength_gram(prev_word)
        str_this = strength_gram(this_word)
        str_next = strength_gram(next_word)

        str_2gram_left = strength_gram(prev_word,this_word) - str_prev - str_this
        str_2gram_right = strength_gram(this_word,next_word) - str_this - str_next

        if n == 0 :
            str_2gram_left = (str_this + str_2gram_right)/2
        elif n == len(data_list[:-2]) - 1 :
            str_2gram_right = (str_2gram_left + str_this)/2

        net_strength = (str_2gram_left + str_this + str_2gram_right)/3

        reasoning_str += f"""<span style = "background-color:"""
        reasoning_str += f"""{strength_to_color_string(net_strength,min_str,max_str)}">{this_word}</span>"""
        reasoning_str += f"""<span style = "background-color:"""
        reasoning_str += f"""{strength_to_color_string(str_2gram_right,min_str,max_str)}"> </span>\n"""
        if n % 12 == 11 :
            reasoning_str += "<br>"
    reasoning_str += "</p>"
    return reasoning_str

def sanitize(data_str : bytes) -> str :
    cleaned_str = ""
    for char in data_str.decode() :
        if char == '<' :
            cleaned_str += '&lt;'
        elif char == '>' :
            cleaned_str += '&gt;'
        elif char == '/':
            cleaned_str += '&sol;'
        elif char == '\\':
            cleaned_str += '&bsol;'
        else :
            cleaned_str += char
    return cleaned_str

def load_model():
    """Load the stored inference model and vocabulary"""
    global vocab
    global model

    with open('vocabulary_hashed.pkl', 'rb') as f_v:
        vocab = pickle.load(f_v)
    with open('sentiment_inference_model_hashed.pkl', 'rb') as f_m:
        model = pickle.load(f_m)


@app.route('/')
def home_endpoint():
    """Locally: what to show when visiting localhost:80"""
    return header+last_return

def _build_cors_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")  # in production, only include mattingliswhalen.github.io
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response

def _corsify_actual_response(data_str):
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.set_data(data_str)
    print(f"Response out: {response.get_data().decode()}")
    return response

@app.route('/predict', methods=['POST','OPTIONS'])
def get_prediction():
    """Locally: what to show when receiving a post request at localhost:80/predict"""
    """>  curl.exe -X POST localhost:80/predict -H 'Content-Type: application/json' -d 'This is a review' """
    global last_return
    # Works only for a single sample
    if request.method == 'OPTIONS' :  # CORS preflight
        return _build_cors_preflight_response()
    elif request.method == 'POST':
        data = sanitize(request.get_data())  # Get data posted as a string
        transformed_data = vocab.transform([data])  # Transform the input string with the HasheVectorizer
        sentiment = model.predict_proba(transformed_data)[0,1]  # Runs globally-loaded model on the data
        last_return = prob_to_html(sentiment) + reasoning_html_from_string(data)
    else :
        raise RuntimeError(f"Can't handle >{request.method}< request method")

    return _corsify_actual_response(last_return)


if __name__ == '__main__':
    load_model()  # load model at the beginning once only
    app.run(host='0.0.0.0', port=80)

