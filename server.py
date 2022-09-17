from flask import Flask, request, jsonify
from classification import classify
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def main_page():
    if request.method == "POST":
        content = request.form["text"]
        y = classify(content, "dataset.txt")
        return jsonify({"class": y})
    return "Hello World"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
