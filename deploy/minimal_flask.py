import pickle

from flask import Flask, request

app = Flask(__name__)

header = """<h1> Minimal Flask </h1>"""
last_return = ""

def load_model() :
    """Load an inference model"""
    global last_return
    last_return = "<p> Dummy model loaded </p>"

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
        this_return = ""
        for word in data.split() :
            this_return += f"{word.decode()} "
        last_return = this_return
    return last_return


if __name__ == '__main__':
    load_model()  # load model at the beginning once only
    app.run(host='0.0.0.0', port=80)

