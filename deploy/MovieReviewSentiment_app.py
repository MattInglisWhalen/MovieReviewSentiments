import pickle
from flask import Flask, request
import re as regex


if __name__ == "__main__" :
    app = Flask(__name__)
    mrs_dir = ""
else :
    from __main__ import app
    mrs_dir = "deploy_mrs"


mrs_vocab = None  # HashingVectorizer
mrs_model = None  # LogisticRegression
mrs_header = ""

EMPTY_STAR = "✰"
FILLED_STAR = "★"

def mrs_prob_to_html(r: float) -> str :
    """Presents the model's predicted probability as html markup"""
    return_str = "<p> Predicted stars: "
    for n in range(10) :
        if n/10 + 0.05 < r :
            return_str += FILLED_STAR
        else :
            return_str += EMPTY_STAR

    return_str += "</p> <p>Sentiment: "
    if r < 0.25 :
        return_str += "Bad"
    elif r > 0.75 :
        return_str += "Good"
    else :
        return_str += "Neutral"
    return_str += "</p>"
    return return_str

def mrs_strength_to_color_string(strength: float, min_str: float, max_str: float) -> str :
    """Returns a color choice as a string in the format #XXYYZZ.
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

def mrs_strength_gram(*words) :
    """Turns a list of words into a strength based on logistic regression coefficient of all possible n-grams"""
    bag_of_words = mrs_vocab.fit_transform([' '.join(list(words))])

    # the following will give 3 entries: 1 for the 2 1-grams and 1 for the 2-gram
    _, indices = bag_of_words.nonzero()
    net_strength = 0
    for idx in indices :
        bag_val = bag_of_words[0,idx]
        sign = bag_val / abs(bag_val) if abs(bag_val) > 1e-10 else 0
        net_strength += sign * mrs_model.coef_[0, idx]

    return net_strength

def mrs_reasoning_html_from_string(data: str) -> str :
    """Returns html markup containing the data string with coloring based on model coefficients"""
    reasoning_str = "<p> Reasoning: "
    # min_str, max_str = min(model.coef_[0]), max(model.coef_[0])
    min_str, max_str = -10, 10

    # pad the data with meaningless buffer strings
    data_list = ("BBBB " + data + " BBBB").split()

    for n, (prev_word, this_word, next_word) in enumerate( zip(data_list[:-2],data_list[1:-1],data_list[2:])):

        str_prev = mrs_strength_gram(prev_word)
        str_this = mrs_strength_gram(this_word)
        str_next = mrs_strength_gram(next_word)

        str_2gram_left = mrs_strength_gram(prev_word, this_word) - str_prev - str_this
        str_2gram_right = mrs_strength_gram(this_word, next_word) - str_this - str_next

        if n == 0 :
            str_2gram_left = (str_this + str_2gram_right)/2
        elif n == len(data_list[:-2]) - 1 :
            str_2gram_right = (str_2gram_left + str_this)/2

        net_strength = (str_2gram_left + str_this + str_2gram_right)/3

        reasoning_str += f"""<span style = "background-color:"""
        reasoning_str += f"""{mrs_strength_to_color_string(net_strength, min_str, max_str)}">{this_word}</span>"""
        reasoning_str += f"""<span style = "background-color:"""
        reasoning_str += f"""{mrs_strength_to_color_string(str_2gram_right, min_str, max_str)}"> </span>\n"""
        if n % 12 == 11 :
            reasoning_str += "<br>"
    reasoning_str += "</p>"
    return reasoning_str

def mrs_sanitize(data_str : bytes) -> str :
    """Cleans the user-input raw text to escape nefarious actions"""
    cleaned_str = ""

    word_str = data_str.decode()
    first_issue = r'\\n'
    second_issue = "\\\\\""
    word_str = regex.sub(first_issue,' <br> ',word_str)
    word_str = regex.sub(second_issue,'&lsquo;',word_str)

    for paragraph in word_str.split('\n') :
        for word in paragraph.split() :
            skip_next = False
            shifted_word = word + ' '
            for char, next_char in zip(word,shifted_word[1:]) :
                if skip_next :
                    skip_next = False
                    continue
                if char == '<' :
                    cleaned_str += ' &lt; '
                elif char == '>' :
                    cleaned_str += ' &gt; '
                elif char == '/':
                    cleaned_str += ' &sol; '
                elif char == '\\' :
                    if next_char == 'n' :
                        cleaned_str += " <br> "
                        skip_next = True
                        continue
                    elif next_char == '"' :
                        cleaned_str += "&ldquo;"
                        skip_next = True
                        continue
                    cleaned_str += ' &bsol; '
                elif char in ['\n','\r']:
                    cleaned_str += ' <br> '
                elif char == '"' :
                    cleaned_str += '&lsquo;'
                else :
                    cleaned_str += char
            cleaned_str += ' '
        cleaned_str += "<br>"

    return cleaned_str

def mrs_load_frontend():
    """To be used on boot; loads header from html"""
    global mrs_header
    with open(mrs_dir+'MovieReviewSentiment.html', 'r') as f_html:
        mrs_header = f_html.read()

def mrs_load_model():
    """Load the stored inference model and vocabulary"""
    global mrs_vocab
    global mrs_model

    with open(mrs_dir+'vocabulary_hashed.pkl', 'rb') as f_v:
        mrs_vocab = pickle.load(f_v)
    with open(mrs_dir+'sentiment_inference_model_hashed.pkl', 'rb') as f_m:
        mrs_model = pickle.load(f_m)


if __name__ == "__main__" :
    @app.route('/')
    def home_endpoint():
        """Locally: what to show when visiting localhost:80"""
        return """Please visit <a href="localhost:80/mrs_demo" target="_blank">
                  the Movie Review Sentiment demo page</a>"""


@app.route('/mrs_demo')
def mrs_demo():
    """Locally: what to show when visiting localhost:80"""
    return mrs_header

# ssh -i C:\Users\Matt\Documents\AWS\AWS_DEPLOYED_MODELS.pem ec2-user@18.216.26.152
# scp -i C:\Users\Matt\Documents\AWS\AWS_DEPLOYED_MODELS.pem files ec2-user@18.216.26.152:/home/ec2-user
@app.route('/mrs_demo/request', methods=['POST','OPTIONS'])
def mrs_demo_prediction():
    """Locally: what to show when receiving a post request at localhost:80/predict"""
    # Usage:
    # >  curl.exe -X POST localhost:80/predict -H 'Content-Type: application/json' -d 'This is a review'

    # Works only for a single sample
    if request.method == 'POST':
        data = mrs_sanitize(request.get_data())  # Get data posted as a string
        transformed_data = mrs_vocab.transform([data])  # Transform the input string with the HashedVectorizer
        sentiment = mrs_model.predict_proba(transformed_data)[0, 1]  # Runs globally-loaded model on the data
        return mrs_prob_to_html(sentiment) + mrs_reasoning_html_from_string(data)
    else :
        raise RuntimeError(f"Can't handle >{request.method}< request method")


if __name__ == '__main__':
    mrs_load_frontend()  # load html for the user-facing site
    mrs_load_model()  # load model at the beginning once only
    app.run(host='0.0.0.0', port=80)

