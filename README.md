# 🎵 TikTok Creator PA

> AI-powered personal assistant for TikTok creators — generate viral captions, hooks, hashtags, content calendars, scripts, and trend analysis using Groq's free LLM API.

## ✨ Features

| Tab | What it does |
|-----|-------------|
| ✍️ Caption Generator | Scroll-stopping captions tailored to your niche and tone |
| 🔁 Caption Rewriter | 3 distinct rewrites of your existing caption |
| 🪝 Hook Generator | 6 opening hooks: controversy, curiosity, story, stat, relatable |
| 🏷️ Hashtag Strategy | Trending · Niche · Hidden gem hashtags + ready-to-copy combo |
| 📅 Content Calendar | Full 7-day posting plan with concepts, hooks, hashtags, and goals |
| 🎬 Script Writer | Word-for-word TikTok scripts with B-roll notes and pacing tips |
| ⚡ A/B Caption Tester | Score two captions across 6 dimensions and pick a winner |
| 🔍 Competitor Audit | Analyse a creator's strategy and find your differentiation angle |
| 📈 Trend Explorer | Live Google Trends chart for up to 5 keywords via pytrends |
| 📊 Engagement Analyser | Score your post metrics vs niche benchmark + AI advice |

## 🚀 Quick Start

```bash
pip install gradio groq pytrends pandas matplotlib seaborn wordcloud pillow
export GROQ_API_KEY="gsk_..."
python tiktok_creator_pa.py
```

Open http://localhost:7861 — paste your API key in the Settings box if you did not set the env var.

## 🛠 Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Groq API — Llama 3.3 70B Versatile (free tier) |
| UI | Gradio 4.x |
| Trend data | pytrends (Google Trends, no key required) |
| Visualisation | Matplotlib · Seaborn · WordCloud |

## 📊 Analysis Notebook

tiktok_analysis.ipynb contains a full exploratory analysis:
- EDA — views, likes, comments, shares, saves, engagement rate distributions
- Hashtag frequency analysis — word cloud + optimal hashtag count
- Google Trends — live search interest for creator keywords
- Caption length vs engagement — scatter + trend line
- Posting time heatmap — best hour and day-of-week
- Niche performance heatmap — 10 niches x 6 metrics
- Caption sentiment vs engagement — DistilBERT sentiment analysis
- Video duration analysis — 15s / 30s / 60s / 90s / 180s comparison

## ⚠️ Limitations

- Generated captions, hooks, and scripts come from a general-purpose LLM (Llama 3.3 70B) — treat them as strong drafts to edit, not guaranteed-viral content.
- pytrends is an unofficial Google Trends client and can be rate-limited, so the Trend Explorer may fail intermittently.
- Engagement scoring compares against static niche benchmarks rather than live TikTok data (no official TikTok API integration).

## 🎯 Why I Built This

TikTok's algorithm rewards consistency and quality — but most solo creators struggle with writing hooks, researching hashtags, and planning content every day. This tool automates the repetitive parts so you can focus on filming.

Built as part of my data analysis / ML portfolio targeting TikTok & Spotify internships.

*University of Hull · 2026*
