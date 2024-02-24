from flask import Flask, render_template

RASA_API_URL = 'http://localhost:5005/webhooks/rest/webhook'
app = Flask(__name__)

if __name__ == "__name__":
    app.run(debug=False, port=3000)