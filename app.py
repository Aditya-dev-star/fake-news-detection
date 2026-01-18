from flask import Flask, render_template, request
from src.predict import predict_news
import requests

app = Flask(__name__)

NEWS_API_KEY = "YOUR_NEWSAPI_KEY"

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    searched_news = []

    # ----- ANALYZE PASTED NEWS -----
    if request.method == "POST" and "news" in request.form:
        news_text = request.form.get("news", "").strip()
        if news_text:
            result = predict_news(news_text)

    # ----- SEARCH LIVE NEWS -----
    if request.method == "POST" and "query" in request.form:
        query = request.form.get("query", "").strip()

        if query:
            url = (
                "https://newsapi.org/v2/everything?"
                f"q={query}&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
            )
            response = requests.get(url).json()

            if "articles" in response:
                for article in response["articles"][:5]:
                    content = (
                        (article.get("title") or "") + " " +
                        (article.get("description") or "")
                    ).strip()

                    if content:
                        label, confidence = predict_news(content)
                        searched_news.append((content, label, confidence))

    return render_template(
        "index.html",
        result=result,
        searched_news=searched_news
    )

if __name__ == "__main__":
    app.run(debug=True)



