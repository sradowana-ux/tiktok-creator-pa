"""
TikTok Creator PA — AI-Powered Personal Assistant for Creators
==============================================================
Run:  python tiktok_creator_pa.py
Open: http://localhost:7860

Requirements:
    pip install gradio groq pytrends pandas matplotlib seaborn wordcloud pillow

Set your API key:
    export GROQ_API_KEY="gsk_..."
    OR paste it directly into the API key box in the app UI
"""

import os
import re
import json
import random
from datetime import datetime, timedelta

import gradio as gr
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from groq import Groq

# ── Optional: pytrends (soft dependency) ─────────────────────────────────
try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except ImportError:
    PYTRENDS_AVAILABLE = False

# ── Config ────────────────────────────────────────────────────────────────
DEFAULT_MODEL   = "llama-3.3-70b-versatile"
FALLBACK_MODEL  = "llama3-8b-8192"
MAX_TOKENS      = 1024
TEMPERATURE     = 0.85

NICHE_OPTIONS = [
    "Fashion & Style", "Beauty & Makeup", "Fitness & Gym",
    "Food & Cooking", "Travel & Lifestyle", "Comedy & Skits",
    "Tech & Gaming", "Finance & Money", "Mental Health & Wellness",
    "Dance & Music", "Motivation & Self-help", "Study & Education",
    "Art & Creativity", "Pets & Animals", "Sports",
]

TONE_OPTIONS = ["Trendy & Gen-Z", "Professional", "Funny & Sarcastic",
                "Inspirational", "Casual & Friendly", "Mysterious & Cryptic"]

EMOJI_MAP = {
    "Fashion & Style": "👗", "Beauty & Makeup": "💄", "Fitness & Gym": "💪",
    "Food & Cooking": "🍳", "Travel & Lifestyle": "✈️", "Comedy & Skits": "😂",
    "Tech & Gaming": "🎮", "Finance & Money": "💰", "Mental Health & Wellness": "🧠",
    "Dance & Music": "🎵", "Motivation & Self-help": "🔥", "Study & Education": "📚",
    "Art & Creativity": "🎨", "Pets & Animals": "🐾", "Sports": "⚽",
}

# ── Groq client ───────────────────────────────────────────────────────────
_client_cache = {}

def get_client(api_key: str) -> Groq:
    key = api_key.strip()
    if key not in _client_cache:
        _client_cache[key] = Groq(api_key=key)
    return _client_cache[key]


def groq_chat(api_key: str, system: str, user: str, temperature: float = TEMPERATURE) -> str:
    client = get_client(api_key)
    for model in [DEFAULT_MODEL, FALLBACK_MODEL]:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                max_tokens=MAX_TOKENS,
                temperature=temperature,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if "model" in str(e).lower():
                continue
            raise e
    raise RuntimeError("Both Groq models failed. Check your API key.")


# ══════════════════════════════════════════════════════════════════════════
#  FEATURE 1 — Caption Generator
# ══════════════════════════════════════════════════════════════════════════
def generate_caption(api_key, description, niche, tone, include_cta, include_hashtags, n_hashtags):
    if not api_key.strip():
        return "⚠️ Please enter your Groq API key in the Settings tab."
    if not description.strip():
        return "⚠️ Please describe your video."

    cta_note  = "End with a strong call-to-action (like, follow, comment)." if include_cta else ""
    hash_note = f"Add {n_hashtags} relevant hashtags at the end." if include_hashtags else "Do NOT include hashtags."
    emoji_hint = EMOJI_MAP.get(niche, "")

    system = (
        f"You are an expert TikTok content strategist specialising in {niche} content. "
        f"You write viral, platform-native captions that feel authentic and drive engagement. "
        f"Tone: {tone}. Use TikTok language and trends where appropriate."
    )
    user = (
        f"Write a TikTok caption for this video:\n\n{description}\n\n"
        f"Niche: {niche} {emoji_hint}\n"
        f"Requirements:\n"
        f"- Hook in the first line (make people stop scrolling)\n"
        f"- Max 150 words\n"
        f"- {cta_note}\n"
        f"- {hash_note}\n"
        f"Return ONLY the caption, no explanation."
    )
    return groq_chat(api_key, system, user)


# ══════════════════════════════════════════════════════════════════════════
#  FEATURE 2 — Caption Rewriter (3 versions)
# ══════════════════════════════════════════════════════════════════════════
def rewrite_caption(api_key, original_caption, niche, improvement_goal):
    if not api_key.strip():
        return "⚠️ Please enter your Groq API key."
    if not original_caption.strip():
        return "⚠️ Paste your existing caption first."

    system = (
        "You are a viral TikTok content expert. You rewrite weak captions into "
        "scroll-stopping, engagement-driving versions."
    )
    user = (
        f"Rewrite this TikTok caption 3 different ways. Goal: {improvement_goal}.\n"
        f"Niche: {niche}\n\n"
        f"Original caption:\n{original_caption}\n\n"
        f"Return exactly 3 versions labelled:\n"
        f"**Version 1 — [style]:**\n[caption]\n\n"
        f"**Version 2 — [style]:**\n[caption]\n\n"
        f"**Version 3 — [style]:**\n[caption]\n\n"
        f"Each version should feel distinctly different. Include hashtags."
    )
    return groq_chat(api_key, system, user)


# ══════════════════════════════════════════════════════════════════════════
#  FEATURE 3 — Hook Generator
# ══════════════════════════════════════════════════════════════════════════
def generate_hooks(api_key, topic, niche, hook_style):
    if not api_key.strip():
        return "⚠️ Please enter your Groq API key."
    if not topic.strip():
        return "⚠️ Please describe your video topic."

    system = (
        "You are a TikTok hook specialist. You write the first 1-2 sentences of TikTok videos "
        "that stop the scroll within 0.5 seconds. You understand pattern interrupts, curiosity gaps, "
        "and psychological triggers that keep viewers watching."
    )
    user = (
        f"Generate 6 different opening hooks for a TikTok video about:\n'{topic}'\n\n"
        f"Niche: {niche}\n"
        f"Hook style preference: {hook_style}\n\n"
        f"Hook types to cover:\n"
        f"1. Controversy / Hot take\n"
        f"2. Curiosity gap ('You won't believe...')\n"
        f"3. Direct address ('If you are X, watch this')\n"
        f"4. Story opener ('I spent 30 days...')\n"
        f"5. Shocking stat or fact\n"
        f"6. Relatable pain point\n\n"
        f"Format: **Hook [N] — [Type]:**\n[hook text]\n\n"
        f"Keep each hook under 15 words. Make them punchy."
    )
    return groq_chat(api_key, system, user)


# ══════════════════════════════════════════════════════════════════════════
#  FEATURE 4 — Hashtag Recommender
# ══════════════════════════════════════════════════════════════════════════
def recommend_hashtags(api_key, video_description, niche, strategy):
    if not api_key.strip():
        return "⚠️ Please enter your Groq API key."

    system = (
        "You are a TikTok SEO and hashtag strategy expert. You know which hashtags "
        "drive reach vs engagement, and how to balance niche and broad hashtags."
    )
    user = (
        f"Recommend a hashtag strategy for this TikTok video:\n\n{video_description}\n\n"
        f"Niche: {niche}\n"
        f"Strategy: {strategy}\n\n"
        f"Provide:\n"
        f"**🔥 Trending (broad reach)** — 5 hashtags with estimated view counts\n"
        f"**🎯 Niche (targeted)** — 5 hashtags specific to {niche}\n"
        f"**💎 Hidden gem (low competition)** — 5 underused hashtags that creators overlook\n"
        f"**📋 Ready-to-copy combo** — your recommended mix of all three (15 hashtags total)\n\n"
        f"Explain briefly WHY each category works differently."
    )
    return groq_chat(api_key, system, user)


# ══════════════════════════════════════════════════════════════════════════
#  FEATURE 5 — Content Calendar Generator
# ══════════════════════════════════════════════════════════════════════════
def generate_calendar(api_key, niche, posting_frequency, content_pillars, upcoming_events):
    if not api_key.strip():
        return "⚠️ Please enter your Groq API key."

    today = datetime.now().strftime("%A %d %B %Y")
    system = (
        "You are a TikTok content strategist who creates data-driven content calendars "
        "that balance entertainment, education, and promotion."
    )
    user = (
        f"Create a 7-day TikTok content calendar starting from {today}.\n\n"
        f"Creator niche: {niche}\n"
        f"Posting frequency: {posting_frequency} posts/day\n"
        f"Content pillars: {content_pillars}\n"
        f"Upcoming events/trends to leverage: {upcoming_events if upcoming_events else 'None specified'}\n\n"
        f"For each post include:\n"
        f"- 📅 Day & best posting time\n"
        f"- 🎬 Video concept (1 sentence)\n"
        f"- 🪝 Opening hook\n"
        f"- 🏷️ 5 hashtags\n"
        f"- 🎯 Goal (reach / engagement / followers / sales)\n\n"
        f"Format as a clean table then detailed breakdown below."
    )
    return groq_chat(api_key, system, user, temperature=0.7)


# ══════════════════════════════════════════════════════════════════════════
#  FEATURE 6 — A/B Caption Tester
# ══════════════════════════════════════════════════════════════════════════
def ab_test_captions(api_key, caption_a, caption_b, niche, target_audience):
    if not api_key.strip():
        return "⚠️ Please enter your Groq API key."
    if not caption_a.strip() or not caption_b.strip():
        return "⚠️ Please enter both captions."

    system = (
        "You are a TikTok performance analyst with deep knowledge of what drives "
        "engagement, saves, shares, and follows on the platform. You score content objectively."
    )
    user = (
        f"Compare these two TikTok captions and predict which performs better.\n\n"
        f"Niche: {niche}\nTarget audience: {target_audience}\n\n"
        f"**Caption A:**\n{caption_a}\n\n"
        f"**Caption B:**\n{caption_b}\n\n"
        f"Score each caption (out of 10) on:\n"
        f"- Hook strength\n"
        f"- Clarity & readability\n"
        f"- Emotional trigger\n"
        f"- CTA effectiveness\n"
        f"- Hashtag quality\n"
        f"- Overall predicted engagement\n\n"
        f"Give a clear WINNER with reasoning, then suggest how to combine the best of both "
        f"into a final optimised version."
    )
    return groq_chat(api_key, system, user, temperature=0.5)


# ══════════════════════════════════════════════════════════════════════════
#  FEATURE 7 — Competitor Audit
# ══════════════════════════════════════════════════════════════════════════
def audit_competitor(api_key, creator_handle, top_videos_desc, niche):
    if not api_key.strip():
        return "⚠️ Please enter your Groq API key."
    if not top_videos_desc.strip():
        return "⚠️ Describe the competitor's top videos."

    system = (
        "You are a TikTok competitive intelligence analyst. You break down what makes "
        "creators successful and identify gaps and opportunities for competing creators."
    )
    user = (
        f"Audit this TikTok creator's content strategy:\n\n"
        f"Creator: @{creator_handle}\n"
        f"Niche: {niche}\n"
        f"Their top performing videos:\n{top_videos_desc}\n\n"
        f"Provide:\n"
        f"**📊 Content Patterns** — What formats/topics they repeat\n"
        f"**💪 Strengths** — Why their content works\n"
        f"**🕳️ Gaps** — What topics they're missing (your opportunity)\n"
        f"**🎯 Differentiation strategy** — Exactly how you should position differently\n"
        f"**📋 3 video ideas** to outperform them this week\n"
    )
    return groq_chat(api_key, system, user, temperature=0.6)


# ══════════════════════════════════════════════════════════════════════════
#  FEATURE 8 — Trend Explorer (pytrends + chart)
# ══════════════════════════════════════════════════════════════════════════
def explore_trends(api_key, keywords_raw, timeframe):
    keywords = [k.strip() for k in keywords_raw.split(",") if k.strip()][:5]
    if not keywords:
        return None, "⚠️ Enter at least one keyword."

    tf_map = {
        "Last 7 days": "now 7-d",
        "Last 30 days": "today 1-m",
        "Last 90 days": "today 3-m",
        "Last 12 months": "today 12-m",
    }
    tf = tf_map.get(timeframe, "now 7-d")

    if not PYTRENDS_AVAILABLE:
        # Simulate data if pytrends not installed
        dates = pd.date_range(end=datetime.now(), periods=30, freq="D")
        df = pd.DataFrame({kw: [random.randint(20, 100) for _ in dates] for kw in keywords}, index=dates)
        note = "(Simulated data — install pytrends for live Google Trends)"
    else:
        try:
            pt = TrendReq(hl="en-US", tz=0)
            pt.build_payload(keywords, timeframe=tf, geo="")
            df = pt.interest_over_time()
            if df.empty:
                return None, "No trend data found. Try different keywords."
            if "isPartial" in df.columns:
                df = df.drop(columns=["isPartial"])
            note = "Source: Google Trends"
        except Exception as e:
            return None, f"Trends fetch failed: {e}"

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.patch.set_facecolor("#0f0f0f")
    colors = ["#fe2c55", "#25f4ee", "#ffffff", "#fffc00", "#69c9d0"]

    # Line chart
    ax1 = axes[0]
    ax1.set_facecolor("#1a1a1a")
    for i, kw in enumerate(df.columns):
        ax1.plot(df.index, df[kw], color=colors[i % len(colors)],
                 linewidth=2.5, label=kw, marker="o", markersize=3)
    ax1.set_title("📈 Search Interest Over Time", color="white", fontsize=13, pad=10)
    ax1.tick_params(colors="white")
    ax1.spines[:].set_color("#333")
    ax1.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=9)
    ax1.set_ylabel("Interest (0–100)", color="white")

    # Bar chart — average interest
    ax2 = axes[1]
    ax2.set_facecolor("#1a1a1a")
    avgs = df.mean().sort_values(ascending=True)
    bars = ax2.barh(avgs.index, avgs.values,
                    color=[colors[i % len(colors)] for i in range(len(avgs))])
    ax2.set_title("📊 Average Interest Comparison", color="white", fontsize=13, pad=10)
    ax2.tick_params(colors="white")
    ax2.spines[:].set_color("#333")
    ax2.set_xlabel("Average Interest", color="white")
    for bar, val in zip(bars, avgs.values):
        ax2.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}", va="center", color="white", fontsize=9)

    plt.tight_layout(pad=2)
    fig.text(0.99, 0.01, note, ha="right", color="#666", fontsize=8)

    return fig, f"✅ Trend data loaded for: {', '.join(keywords)}"


# ══════════════════════════════════════════════════════════════════════════
#  FEATURE 9 — Engagement Score Analyser
# ══════════════════════════════════════════════════════════════════════════
def analyse_engagement(api_key, views, likes, comments, shares, saves, followers, niche):
    try:
        v, l, c, sh, sv, f = (int(x.replace(",", "").replace(" ", "")) if x.strip()
                               else 0 for x in [views, likes, comments, shares, saves, followers])
    except ValueError:
        return None, "⚠️ Enter numbers only in the metric fields."

    if v == 0:
        return None, "⚠️ Views cannot be zero."

    # Rates
    like_rate    = l / v * 100
    comment_rate = c / v * 100
    share_rate   = sh / v * 100
    save_rate    = sv / v * 100
    total_er     = (l + c + sh + sv) / v * 100

    # TikTok benchmarks by niche
    benchmarks = {
        "Fashion & Style": 6.0, "Beauty & Makeup": 7.0, "Fitness & Gym": 8.0,
        "Food & Cooking": 7.5, "Travel & Lifestyle": 5.5, "Comedy & Skits": 9.0,
        "Tech & Gaming": 5.0, "Finance & Money": 4.5, "Mental Health & Wellness": 8.5,
        "Dance & Music": 10.0, "Motivation & Self-help": 7.0, "Study & Education": 6.0,
        "Art & Creativity": 7.5, "Pets & Animals": 9.5, "Sports": 6.5,
    }
    benchmark = benchmarks.get(niche, 6.5)
    score_pct  = min(total_er / benchmark * 100, 150)

    if score_pct >= 120:  grade, colour = "🏆 Viral", "#fe2c55"
    elif score_pct >= 90: grade, colour = "🔥 Great",  "#fffc00"
    elif score_pct >= 65: grade, colour = "✅ Good",   "#25f4ee"
    elif score_pct >= 40: grade, colour = "📈 Average","#69c9d0"
    else:                  grade, colour = "📉 Below average", "#888"

    # Donut chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    fig.patch.set_facecolor("#0f0f0f")

    metrics = {"Likes": like_rate, "Comments": comment_rate,
               "Shares": share_rate, "Saves": save_rate}
    mc = ["#fe2c55", "#25f4ee", "#fffc00", "#69c9d0"]
    wedges, texts, autotexts = ax1.pie(
        list(metrics.values()), labels=list(metrics.keys()),
        colors=mc, autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": "#0f0f0f", "linewidth": 2},
        textprops={"color": "white"},
    )
    for at in autotexts: at.set_color("white")
    ax1.set_facecolor("#0f0f0f")
    ax1.set_title("Engagement Breakdown", color="white", fontsize=12)

    # Score gauge (bar)
    ax2.set_facecolor("#1a1a1a")
    ax2.barh(["Your ER", f"{niche} avg"], [total_er, benchmark],
             color=[colour, "#444"], height=0.4)
    ax2.set_title(f"Engagement Rate vs Benchmark\n{grade}", color="white", fontsize=12)
    ax2.tick_params(colors="white")
    ax2.spines[:].set_color("#333")
    ax2.set_xlabel("Engagement Rate (%)", color="white")
    for i, val in enumerate([total_er, benchmark]):
        ax2.text(val + 0.05, i, f"{val:.2f}%", va="center", color="white", fontsize=10)
    plt.tight_layout()

    summary = (
        f"**{grade}**  |  Total ER: **{total_er:.2f}%**  |  Benchmark: {benchmark}%\n\n"
        f"| Metric | Count | Rate |\n|--------|-------|------|\n"
        f"| 👍 Likes | {l:,} | {like_rate:.2f}% |\n"
        f"| 💬 Comments | {c:,} | {comment_rate:.2f}% |\n"
        f"| 🔁 Shares | {sh:,} | {share_rate:.2f}% |\n"
        f"| 🔖 Saves | {sv:,} | {save_rate:.2f}% |\n"
    )

    if api_key.strip():
        try:
            advice = groq_chat(
                api_key,
                "You are a TikTok growth strategist.",
                f"My TikTok post in {niche} got {v:,} views, ER {total_er:.2f}% vs benchmark {benchmark}%. "
                f"Likes {like_rate:.2f}%, Comments {comment_rate:.2f}%, Shares {share_rate:.2f}%, Saves {save_rate:.2f}%. "
                f"Give me 3 specific, actionable things to improve my next video. Be direct.",
                temperature=0.6,
            )
            summary += f"\n\n---\n**🤖 AI Advice:**\n{advice}"
        except Exception:
            pass

    return fig, summary


# ══════════════════════════════════════════════════════════════════════════
#  FEATURE 10 — Video Script Writer
# ══════════════════════════════════════════════════════════════════════════
def write_script(api_key, topic, niche, duration, cta_goal):
    if not api_key.strip():
        return "⚠️ Please enter your Groq API key."
    if not topic.strip():
        return "⚠️ Please describe your video topic."

    system = (
        "You are an expert TikTok scriptwriter who creates short-form scripts that feel "
        "natural, keep retention high, and drive action. You understand pacing, pattern "
        "interrupts, and the 'retention curve' of TikTok videos."
    )
    user = (
        f"Write a full TikTok video script for:\n'{topic}'\n\n"
        f"Niche: {niche}\n"
        f"Target duration: {duration} seconds\n"
        f"CTA goal: {cta_goal}\n\n"
        f"Format:\n"
        f"**🪝 HOOK (0–3s):** [exact words to say]\n"
        f"**📖 BODY:** [broken into 10-15s segments with [B-ROLL] notes]\n"
        f"**🎯 CTA (last 5s):** [exact words]\n"
        f"**📝 CAPTION:** [ready-to-copy caption + hashtags]\n"
        f"**⏱️ PACING NOTES:** [tips on delivery and cuts]\n\n"
        f"Write the script word-for-word as if the creator will read it directly."
    )
    return groq_chat(api_key, system, user)


# ══════════════════════════════════════════════════════════════════════════
#  GRADIO UI
# ══════════════════════════════════════════════════════════════════════════
CSS = """
body, .gradio-container { background: #0f0f0f !important; color: white !important; }
.gr-button-primary { background: #fe2c55 !important; border: none !important; }
.gr-button { background: #1a1a1a !important; color: white !important; border: 1px solid #333 !important; }
.gr-input, .gr-textarea, .gr-dropdown select { background: #1a1a1a !important; color: white !important; border: 1px solid #333 !important; }
.gr-panel { background: #111 !important; border: 1px solid #222 !important; }
h1, h2, h3 { color: white !important; }
.gr-markdown { color: #ccc !important; }
label { color: #aaa !important; }
"""

HEADER = """
<div style="text-align:center; padding:20px 0 10px 0;">
  <h1 style="font-size:2.2rem; font-weight:800; color:white; margin:0;">
    🎵 TikTok Creator PA
  </h1>
  <p style="color:#aaa; margin:6px 0 0 0; font-size:1rem;">
    Your AI-powered personal assistant for viral TikTok content
  </p>
  <div style="margin-top:8px;">
    <span style="background:#fe2c55; color:white; padding:3px 10px; border-radius:12px; font-size:0.8rem; margin:2px;">Captions</span>
    <span style="background:#25f4ee; color:black; padding:3px 10px; border-radius:12px; font-size:0.8rem; margin:2px;">Hooks</span>
    <span style="background:#ffffff; color:black; padding:3px 10px; border-radius:12px; font-size:0.8rem; margin:2px;">Hashtags</span>
    <span style="background:#fffc00; color:black; padding:3px 10px; border-radius:12px; font-size:0.8rem; margin:2px;">Trends</span>
    <span style="background:#69c9d0; color:black; padding:3px 10px; border-radius:12px; font-size:0.8rem; margin:2px;">Analytics</span>
  </div>
</div>
"""

with gr.Blocks(title="TikTok Creator PA", css=CSS) as demo:
    gr.HTML(HEADER)

    # ── Shared API key state ───────────────────────────────────────────
    with gr.Accordion("⚙️ Settings — Groq API Key", open=False):
        api_key_box = gr.Textbox(
            label="Groq API Key",
            placeholder="gsk_...",
            type="password",
            value=os.environ.get("GROQ_API_KEY", ""),
            info="Get yours free at console.groq.com → API Keys",
        )

    with gr.Tabs():

        # ── Tab 1: Caption Generator ───────────────────────────────────
        with gr.Tab("✍️ Caption Generator"):
            with gr.Row():
                with gr.Column():
                    cap_desc     = gr.Textbox(label="Describe your video", lines=4,
                                              placeholder="e.g. I'm showing my 5am morning routine as a uni student...")
                    cap_niche    = gr.Dropdown(NICHE_OPTIONS, label="Your niche", value="Study & Education")
                    cap_tone     = gr.Dropdown(TONE_OPTIONS, label="Tone", value="Trendy & Gen-Z")
                    cap_cta      = gr.Checkbox(label="Include call-to-action", value=True)
                    cap_hashtags = gr.Checkbox(label="Include hashtags", value=True)
                    cap_n_hash   = gr.Slider(5, 30, value=15, step=5, label="Number of hashtags")
                    cap_btn      = gr.Button("🚀 Generate Caption", variant="primary")
                with gr.Column():
                    cap_out = gr.Textbox(label="Your caption", lines=15, show_copy_button=True)
            cap_btn.click(generate_caption,
                          inputs=[api_key_box, cap_desc, cap_niche, cap_tone,
                                  cap_cta, cap_hashtags, cap_n_hash],
                          outputs=cap_out)
            gr.Examples([
                ["I tried eating only protein for 30 days and here's what happened to my body", "Fitness & Gym", "Trendy & Gen-Z"],
                ["POV: you're a broke student trying to cook a £2 meal that actually slaps", "Food & Cooking", "Funny & Sarcastic"],
                ["I asked 100 people on the street what they spend on rent and the answers shocked me", "Finance & Money", "Inspirational"],
            ], inputs=[cap_desc, cap_niche, cap_tone])

        # ── Tab 2: Caption Rewriter ────────────────────────────────────
        with gr.Tab("🔁 Caption Rewriter"):
            with gr.Row():
                with gr.Column():
                    rew_orig  = gr.Textbox(label="Paste your existing caption", lines=6,
                                           placeholder="Your current caption here...")
                    rew_niche = gr.Dropdown(NICHE_OPTIONS, label="Your niche", value="Fitness & Gym")
                    rew_goal  = gr.Dropdown(
                        ["More views", "More comments", "More follows", "More saves", "More shares", "More sales"],
                        label="Improvement goal", value="More views"
                    )
                    rew_btn   = gr.Button("✨ Rewrite 3 Ways", variant="primary")
                with gr.Column():
                    rew_out = gr.Textbox(label="3 rewritten versions", lines=20, show_copy_button=True)
            rew_btn.click(rewrite_caption,
                          inputs=[api_key_box, rew_orig, rew_niche, rew_goal],
                          outputs=rew_out)

        # ── Tab 3: Hook Generator ──────────────────────────────────────
        with gr.Tab("🪝 Hook Generator"):
            with gr.Row():
                with gr.Column():
                    hook_topic = gr.Textbox(label="What is your video about?", lines=3,
                                            placeholder="e.g. How I paid off £5000 debt in 6 months as a student")
                    hook_niche = gr.Dropdown(NICHE_OPTIONS, label="Your niche", value="Finance & Money")
                    hook_style = gr.Dropdown(
                        ["Mix of all styles", "Controversial", "Storytelling", "Curiosity-driven", "Relatable"],
                        label="Preferred hook style", value="Mix of all styles"
                    )
                    hook_btn   = gr.Button("🎣 Generate 6 Hooks", variant="primary")
                with gr.Column():
                    hook_out = gr.Textbox(label="Your hooks", lines=20, show_copy_button=True)
            hook_btn.click(generate_hooks,
                           inputs=[api_key_box, hook_topic, hook_niche, hook_style],
                           outputs=hook_out)

        # ── Tab 4: Hashtag Recommender ────────────────────────────────
        with gr.Tab("🏷️ Hashtags"):
            with gr.Row():
                with gr.Column():
                    hash_desc     = gr.Textbox(label="Describe your video", lines=4,
                                               placeholder="What is your video about?")
                    hash_niche    = gr.Dropdown(NICHE_OPTIONS, label="Your niche", value="Beauty & Makeup")
                    hash_strategy = gr.Dropdown(
                        ["Balanced (reach + niche)", "Max reach (broad)", "Niche authority", "Trending only"],
                        label="Hashtag strategy", value="Balanced (reach + niche)"
                    )
                    hash_btn = gr.Button("🔍 Get Hashtag Strategy", variant="primary")
                with gr.Column():
                    hash_out = gr.Textbox(label="Your hashtag strategy", lines=20, show_copy_button=True)
            hash_btn.click(recommend_hashtags,
                           inputs=[api_key_box, hash_desc, hash_niche, hash_strategy],
                           outputs=hash_out)

        # ── Tab 5: Content Calendar ────────────────────────────────────
        with gr.Tab("📅 Content Calendar"):
            with gr.Row():
                with gr.Column():
                    cal_niche    = gr.Dropdown(NICHE_OPTIONS, label="Your niche", value="Fitness & Gym")
                    cal_freq     = gr.Slider(1, 5, value=2, step=1, label="Posts per day")
                    cal_pillars  = gr.Textbox(label="Your content pillars (comma separated)",
                                              placeholder="e.g. Workouts, Meal prep, Motivation, Day in my life",
                                              value="Education, Entertainment, Inspiration")
                    cal_events   = gr.Textbox(label="Upcoming events / trends to leverage",
                                              placeholder="e.g. Summer, exam season, new Netflix show...")
                    cal_btn      = gr.Button("📆 Generate 7-Day Calendar", variant="primary")
                with gr.Column():
                    cal_out = gr.Textbox(label="Your content calendar", lines=25, show_copy_button=True)
            cal_btn.click(generate_calendar,
                          inputs=[api_key_box, cal_niche, cal_freq, cal_pillars, cal_events],
                          outputs=cal_out)

        # ── Tab 6: Script Writer ───────────────────────────────────────
        with gr.Tab("🎬 Script Writer"):
            with gr.Row():
                with gr.Column():
                    scr_topic    = gr.Textbox(label="Video topic", lines=3,
                                              placeholder="e.g. 5 things I wish I knew before starting uni")
                    scr_niche    = gr.Dropdown(NICHE_OPTIONS, label="Your niche", value="Study & Education")
                    scr_duration = gr.Dropdown(["15", "30", "60", "90", "180"],
                                               label="Target duration (seconds)", value="60")
                    scr_cta      = gr.Dropdown(
                        ["Follow for more", "Comment your answer", "Save this for later",
                         "Share with a friend", "Visit link in bio", "Like if you agree"],
                        label="CTA goal", value="Follow for more"
                    )
                    scr_btn = gr.Button("📝 Write Full Script", variant="primary")
                with gr.Column():
                    scr_out = gr.Textbox(label="Your script", lines=25, show_copy_button=True)
            scr_btn.click(write_script,
                          inputs=[api_key_box, scr_topic, scr_niche, scr_duration, scr_cta],
                          outputs=scr_out)

        # ── Tab 7: A/B Caption Tester ─────────────────────────────────
        with gr.Tab("⚡ A/B Tester"):
            with gr.Row():
                cap_a = gr.Textbox(label="Caption A", lines=6, placeholder="First caption version...")
                cap_b = gr.Textbox(label="Caption B", lines=6, placeholder="Second caption version...")
            with gr.Row():
                ab_niche    = gr.Dropdown(NICHE_OPTIONS, label="Your niche", value="Fitness & Gym")
                ab_audience = gr.Textbox(label="Target audience", placeholder="e.g. 18-24 year old women into fitness")
            ab_btn = gr.Button("⚡ Compare & Score", variant="primary")
            ab_out = gr.Textbox(label="Analysis & winner", lines=15, show_copy_button=True)
            ab_btn.click(ab_test_captions,
                         inputs=[api_key_box, cap_a, cap_b, ab_niche, ab_audience],
                         outputs=ab_out)

        # ── Tab 8: Competitor Audit ────────────────────────────────────
        with gr.Tab("🔍 Competitor Audit"):
            with gr.Row():
                with gr.Column():
                    comp_handle = gr.Textbox(label="Competitor TikTok handle", placeholder="e.g. charlidamelio")
                    comp_niche  = gr.Dropdown(NICHE_OPTIONS, label="Their niche", value="Dance & Music")
                    comp_videos = gr.Textbox(label="Describe their top 3-5 videos",
                                             lines=6,
                                             placeholder="e.g.\n1. 'Day in my life as a NYC dancer' - 2M views\n2. 'Dance tutorial to viral sound' - 5M views\n...")
                    comp_btn    = gr.Button("🔍 Audit Competitor", variant="primary")
                with gr.Column():
                    comp_out = gr.Textbox(label="Competitive intelligence report", lines=20, show_copy_button=True)
            comp_btn.click(audit_competitor,
                           inputs=[api_key_box, comp_handle, comp_videos, comp_niche],
                           outputs=comp_out)

        # ── Tab 9: Trend Explorer ─────────────────────────────────────
        with gr.Tab("📈 Trend Explorer"):
            with gr.Row():
                with gr.Column():
                    tr_keywords  = gr.Textbox(label="Keywords to compare (comma-separated, max 5)",
                                              placeholder="e.g. GRWM, day in my life, vlog, aesthetic, POV")
                    tr_timeframe = gr.Dropdown(
                        ["Last 7 days", "Last 30 days", "Last 90 days", "Last 12 months"],
                        label="Timeframe", value="Last 30 days"
                    )
                    tr_btn = gr.Button("📊 Explore Trends", variant="primary")
                    tr_status = gr.Textbox(label="Status", interactive=False)
                with gr.Column():
                    tr_chart = gr.Plot(label="Trend Chart")
            tr_btn.click(explore_trends,
                         inputs=[api_key_box, tr_keywords, tr_timeframe],
                         outputs=[tr_chart, tr_status])
            gr.Examples([
                ["GRWM, day in my life, vlog, aesthetic, POV"],
                ["tiktok fitness, workout routine, gym motivation, body transformation"],
                ["recipe, cooking, food, meal prep, what I eat in a day"],
            ], inputs=[tr_keywords])

        # ── Tab 10: Engagement Analyser ───────────────────────────────
        with gr.Tab("📊 Engagement Analyser"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Enter your post metrics")
                    eng_views     = gr.Textbox(label="Views", placeholder="e.g. 50000")
                    eng_likes     = gr.Textbox(label="Likes", placeholder="e.g. 3500")
                    eng_comments  = gr.Textbox(label="Comments", placeholder="e.g. 120")
                    eng_shares    = gr.Textbox(label="Shares", placeholder="e.g. 450")
                    eng_saves     = gr.Textbox(label="Saves", placeholder="e.g. 800")
                    eng_followers = gr.Textbox(label="Your followers", placeholder="e.g. 5000")
                    eng_niche     = gr.Dropdown(NICHE_OPTIONS, label="Your niche", value="Fitness & Gym")
                    eng_btn       = gr.Button("📊 Analyse Engagement", variant="primary")
                with gr.Column():
                    eng_chart = gr.Plot(label="Engagement breakdown")
                    eng_out   = gr.Markdown(label="Score & advice")
            eng_btn.click(analyse_engagement,
                          inputs=[api_key_box, eng_views, eng_likes, eng_comments,
                                  eng_shares, eng_saves, eng_followers, eng_niche],
                          outputs=[eng_chart, eng_out])

    gr.Markdown(
        "---\n"
        "**TikTok Creator PA** · Powered by Groq + Llama 3.3 · "
        "Built for the TikTok Creator Tools Portfolio · 2026"
    )

if __name__ == "__main__":
    demo.launch(share=False, server_port=7861)
