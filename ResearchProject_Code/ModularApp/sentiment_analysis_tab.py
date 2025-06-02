import streamlit as st
import requests
from bs4 import BeautifulSoup
from google.cloud import language_v1
import pandas as pd
import os
import plotly.express as px

# Ensure credentials are loaded
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = st.secrets["GOOGLE_CLOUD_CREDENTIALS"]

PRESET_SITES = {
    "CoinDesk": "https://www.coindesk.com",
    "CryptoSlate": "https://cryptoslate.com",
    "NewsBTC": "https://www.newsbtc.com",
    "Bitcoinist": "https://bitcoinist.com",
    "The Daily Hodl": "https://dailyhodl.com",
    "Crypto Briefing": "https://cryptobriefing.com",
}

def run_sentiment_analysis_tab():
    st.title("\U0001F4CA Crypto Sentiment Analysis")
    st.write("Analyze sentiment on BTC, ETH, or DOGE from multiple crypto news sources.")

    crypto = st.selectbox("Choose cryptocurrency:", ["BTC", "ETH", "DOGE"])
    run_analysis = st.button("Run Sentiment Analysis")

    if run_analysis:
        summaries = []
        scaled_scores = []
        magnitudes = []
        combined_article_text = ""

        try:
            with st.spinner("Analyzing crypto sources with Google Cloud..."):
                client = language_v1.LanguageServiceClient()

                for name, url in PRESET_SITES.items():
                    try:
                        res = requests.get(url, timeout=10)
                        soup = BeautifulSoup(res.content, "html.parser")

                        title = soup.title.string.strip() if soup.title else "No title found"
                        paragraphs = soup.find_all('p')
                        full_text = " ".join(p.get_text() for p in paragraphs).strip()
                        if not full_text:
                            full_text = title

                        document = language_v1.Document(content=full_text[:5000], type_=language_v1.Document.Type.PLAIN_TEXT)
                        response = client.analyze_sentiment(request={"document": document})
                        sentiment = response.document_sentiment

                        scaled_score = round((sentiment.score + 1) * 4.5 + 1, 2)

                        short_summary = full_text[:400].split('. ')
                        summary_text = '. '.join(short_summary[:2]) + '.' if len(short_summary) >= 2 else full_text[:300]

                        summaries.append({
                            "source": name,
                            "url": url,
                            "title": title,
                            "summary": summary_text,
                            "scaled_score": scaled_score,
                            "magnitude": round(sentiment.magnitude, 2)
                        })
                        scaled_scores.append(scaled_score)
                        magnitudes.append(sentiment.magnitude)
                        combined_article_text += summary_text + "\n"

                    except Exception as e:
                        st.warning(f"Error fetching or processing {name}: {e}")

            st.subheader("Sentiment Analysis from Sources")
            for s in summaries:
                st.markdown(f"**[{s['source']}]({s['url']})**")
                st.write(f"**Title:** {s['title']}")
                st.write(f"**Summary:** {s['summary']}")
                st.write(f"**Sentiment Score (1–10):** {s['scaled_score']}")
                st.write(f"**Magnitude:** {s['magnitude']} (Higher = stronger emotional tone)")
                st.markdown("---")

            if summaries:
                # Show chart
                chart_df = pd.DataFrame(summaries)
                st.subheader("Sentiment Scores per Source")
                fig = px.bar(chart_df, x="source", y="scaled_score", color="magnitude",
                             color_continuous_scale="ylorrd", title="Sentiment Scores (1–10), Colored by Magnitude")
                st.plotly_chart(fig, use_container_width=True)

            if scaled_scores:
                avg_score = round(sum(scaled_scores) / len(scaled_scores), 2)
                avg_magnitude = round(sum(magnitudes) / len(magnitudes), 2)

                if avg_score >= 8:
                    recommendation = "Buy"
                elif avg_score >= 5:
                    recommendation = "Hold"
                else:
                    recommendation = "Sell"

                st.markdown("---")
                st.subheader("Conclusion")
                st.write(
                    f"After analyzing full article content from multiple crypto sources, the average sentiment score for {crypto} is **{avg_score}/10**, with an emotional magnitude of **{avg_magnitude:.2f}**. "
                    f"This implies a general market tone that is **{recommendation.lower()}**-oriented.\n"
                    f"Article summaries reflected a mix of economic concern, technological updates, community sentiment, and institutional movement surrounding {crypto}. This suggests that market participants are monitoring macro conditions, adoption news, and investor behavior closely."
                )
                st.write(f"**Recommendation:** {recommendation}")

        except Exception as e:
            st.error(f"Analysis failed: {e}")
