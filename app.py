# app.py
import os
import json
import random
from collections import deque
from datetime import datetime

import streamlit as st
import pandas as pd
import duckdb
from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai
from src.etl import load_csvs, build_merged

# -------------------------
# Basic app config & CSS
# -------------------------
st.set_page_config(page_title="Pluto — GenAI E-commerce Assistant", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg,#012a4a,#0077b6); color: #ffffff; }
    .hero { text-align:center; padding:80px 20px; border-radius:12px; margin-bottom:10px; }
    .product-card { background: linear-gradient(90deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02)); padding:14px; border-radius:12px; margin-bottom:10px; }
    .small-muted { color: rgba(255,255,255,0.85); font-size:13px; }
    .user-bubble { background:#0077b6; color:white; padding:10px 14px; border-radius:18px; display:inline-block; margin:6px 0; }
    .bot-bubble { background:#f3f6f9; color:#012a4a; padding:10px 14px; border-radius:18px; display:inline-block; margin:6px 0; }
    .chat-box { background: rgba(255,255,255,0.03); border-radius:10px; padding:12px; max-height:360px; overflow:auto; }
    .analytics-insight { background: rgba(255,255,255,0.04); padding:14px; border-radius:10px; color: #e6f7ff; }
    .stTextInput>div>div>input { background-color: #ffffff !important; color: #012a4a !important; border-radius:6px; }
    .stButton>button { background-color:#00aaff; color:white; border-radius:8px; padding:8px 14px; border:none; }
    .stButton>button:hover { transform: scale(1.02); }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Load Gemini key
# -------------------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("Gemini API key not found. Add GEMINI_API_KEY to your .env file and restart.")
    st.stop()

genai.configure(api_key=API_KEY)

# -------------------------
# Load dataset
# -------------------------
@st.cache_data
def load_data():
    orders, items, products, customers, cat_trans = load_csvs("data/archive/")
    df = build_merged(orders, items, products, customers, cat_trans)
    if "product_category_name_english" in df.columns and "category_en" not in df.columns:
        df = df.rename(columns={"product_category_name_english": "category_en"})
    for c in ["price", "freight_value"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

with st.spinner("Preparing Pluto..."):
    df = load_data()

# -------------------------
# Utilities
# -------------------------
PRODUCT_NAME_TEMPLATES = [
    "Classic {cat}", "Premium {cat}", "Modern {cat}", "Trendy {cat}",
    "{cat} Essentials", "{cat} Bundle", "{cat} Deluxe"
]

def make_display_names(df_slice):
    names = []
    for i, cat in enumerate(df_slice["category_en"].fillna("Unknown")):
        base = str(cat).replace("_", " ").title()
        names.append(random.choice(PRODUCT_NAME_TEMPLATES).format(cat=base) + f" #{i+1}")
    return names

def safe_date(val):
    try:
        dt = pd.to_datetime(val, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.date().isoformat()
    except Exception:
        return None

def parse_query_with_gemini(query_text: str):
    prompt = f"""
You are a shopping assistant that parses user queries into JSON with:
- category: a category name close to dataset field 'category_en' (or empty)
- max_price: numeric maximum price if present else null
Return valid JSON only.

Query: \"{query_text}\"
"""
    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        resp = model.generate_content(prompt)
        txt = resp.text.strip().replace("```json", "").replace("```", "").strip()
        parsed = json.loads(txt)
        return parsed.get("category", "") or "", parsed.get("max_price", None)
    except Exception:
        tokens = [t.lower().strip() for t in query_text.split()]
        cats = df["category_en"].dropna().unique().tolist()
        for t in tokens:
            for c in cats:
                if t in c.lower() or c.lower() in t:
                    return c, None
        return "", None

def nl_to_sql_with_gemini(question: str, schema_columns):
    prompt = f"""
You are an SQL expert. Return only a valid DuckDB SELECT query that answers the question.
Table name: df
Columns: {', '.join(schema_columns)}
Use GROUP BY, ORDER BY, LIMIT 20 where relevant.
Question: {question}
"""
    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        resp = model.generate_content(prompt)
        sql = resp.text.strip().replace("```sql", "").replace("```", "").strip()
        return sql
    except Exception as e:
        return f"-- Gemini error: {e}"

# -------------------------
# Session state
# -------------------------
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "cart" not in st.session_state:
    st.session_state.cart = deque(maxlen=200)
if "last_added" not in st.session_state:
    st.session_state.last_added = None

# -------------------------
# Sidebar navigation
# -------------------------
with st.sidebar:
    st.markdown("## Navigation")
    if st.button(" Home"):
        st.session_state.page = "Home"
        st.rerun()
    if st.button(" Shop (Chat)"):
        st.session_state.page = "Shop (Chat)"
        st.rerun()
    if st.button(" Analytics"):
        st.session_state.page = "Analytics"
        st.rerun()
    st.markdown("---")
    if st.session_state.cart:
        st.markdown("###  Cart")
        cart_df = pd.DataFrame(list(st.session_state.cart))
        st.table(cart_df[["product", "price", "eta", "payment_type"]].rename(
            columns={"product":"Product", "price":"Price", "eta":"Est. Delivery", "payment_type":"Payment"}))
        st.markdown(f"**Total:** ₹{cart_df['price'].sum():.2f}")
        if st.button("Checkout"):
            st.success("✅ Order placed (demo).")
            st.session_state.cart.clear()
            st.session_state.last_added = None

# -------------------------
# HOME
# -------------------------
if st.session_state.page == "Home":
    st.markdown("<div class='hero'><h1> Welcome to Pluto</h1><p style='font-size:18px;color:#e6f7ff;'>Find anything, anytime — your personal GenAI shopping assistant.</p></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        if st.button(" Start Exploring", key="start_explore"):
            st.session_state.page = "Shop (Chat)"
            st.rerun()
    st.markdown("---")
    st.markdown("**Quick actions**")
    st.write("-  Chat with Pluto to look up products.")
    st.write("-  Upload an image to find visually similar products.")
    st.write("-  Analytics: ask business questions in plain English.")
    st.markdown("---")
    st.caption("Keep your GEMINI_API_KEY secret. Do not push it to GitHub.")

# -------------------------
# SHOP (Chat)
# -------------------------
# -------------------------------------------------------
# SHOP (Chat) — Conversational Shopping + Delivery Details
# -------------------------------------------------------
elif st.session_state.page == "Shop (Chat)":
    st.header("🛒 Pluto Shop — Chat & Visual Search")
    st.markdown("Type naturally: *'Hi'*, *'electronics'*, *'Show watches under 500'* or upload an image to find products.*")

    left, right = st.columns([3, 2])

    # -------------------------------
    # 💬 CHATBOT SECTION
    # -------------------------------
    with left:
        st.markdown("####  Conversation")
        st.markdown("<div class='chat-box'>", unsafe_allow_html=True)

        # Display last 30 messages
        for who, text in st.session_state.chat_history[-30:]:
            if who.lower() in ("you", "user"):
                st.markdown(f"<div class='user-bubble'>{text}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='bot-bubble'>{text}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Input box
        user_input = st.text_input("Chat with Pluto", placeholder="e.g., 'Show me dresses under 1000'")

        # When user sends a message
        if st.button("Send") and user_input.strip():
            user_input = user_input.strip()
            st.session_state.chat_history.append(("You", user_input))
            lower_input = user_input.lower()

            # --- Greeting response ---
            if any(word in lower_input for word in ["hi", "hello", "hey"]):
                reply = "Hey ! I'm Pluto. What would you like to shop for today?"
                st.session_state.chat_history.append(("Pluto", reply))

            else:
                # --- Parse category & price from Gemini ---
                category, max_price = parse_query_with_gemini(user_input)

                filtered = df.copy()
                if category:
                    filtered = filtered[filtered["category_en"].str.contains(category, case=False, na=False)]
                if max_price:
                    filtered = filtered[filtered["price"] <= float(max_price)]

                if filtered.empty:
                    st.session_state.chat_history.append(("Pluto", "Hmm, I couldn’t find that category. Try a different keyword like *fashion*, *electronics*, or *beauty*."))
                else:
                    # Create display names and remove duplicates
                    filtered = filtered.drop_duplicates(subset=["product_id"]).reset_index(drop=True)
                    filtered["display_name"] = make_display_names(filtered)

                    st.session_state.chat_history.append(("Pluto", f"Here are some {category.replace('_', ' ').title()} products you might like:"))

                    # --- Display top 6 products with details ---
                    st.markdown("###  Recommended Products")
                    cols = st.columns(3)
                    for i, r in filtered.head(6).iterrows():
                        with cols[i % 3]:
                            name = r["display_name"]
                            price = float(r["price"] or 0)
                            freight = float(r.get("freight_value") or 0)

                            # Sample delivery and payment info
                            delivery_est = safe_date(r.get("order_estimated_delivery_date")) or "5–7 days"
                            handed = safe_date(r.get("order_delivered_carrier_date"))
                            delivered = safe_date(r.get("order_delivered_customer_date"))
                            payment_type = r.get("payment_type") if "payment_type" in r else "Credit Card"

                            # Product Card
                            st.markdown(
                                f"""
                                <div class='product-card'>
                                    <h4>{name}</h4>
                                    <p> ₹{price:.2f}  •   Freight ₹{freight:.2f}</p>
                                    <p> Est. Delivery: {delivery_est}</p>
                                    <p> Payment: {payment_type}</p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            # Buttons
                            if st.button("🛒 Add to Cart", key=f"add_{i}"):
                                st.session_state.cart.append({
                                    "product": name,
                                    "price": price,
                                    "delivery": delivery_est,
                                    "payment": payment_type,
                                })
                                st.success(f"✅ {name} added to cart! Delivery expected in {delivery_est}.")
                            if st.button("ℹ️ View Details", key=f"det_{i}"):
                                st.info(f" {name}\n\n ₹{price:.2f}\n Freight ₹{freight:.2f}\n Delivery: {delivery_est}\n Payment: {payment_type}")

    # -------------------------------
    # 🔍 VISUAL SEARCH SECTION
    # -------------------------------
    with right:
        st.subheader(" Visual Search (optional)")
        uploaded = st.file_uploader("Upload a product image", type=["png", "jpg", "jpeg"])

        if uploaded:
            st.image(uploaded, use_column_width=True)
            img = Image.open(uploaded)
            with st.spinner("Analyzing image with Gemini..."):
                resp = genai.GenerativeModel("models/gemini-2.5-flash").generate_content([
                    "Describe this product for an e-commerce recommendation (category, type, color):",
                    img
                ])
            desc = resp.text.strip()
            st.success(f"Gemini says: {desc}")

            key = desc.split()[0] if desc else ""
            if key:
                matches = df[df["category_en"].str.contains(key, case=False, na=False)]
                if not matches.empty:
                    matches = matches.drop_duplicates(subset=["product_id"]).head(6)
                    matches["display_name"] = make_display_names(matches)
                    st.write("###  Matching Products")
                    cols = st.columns(3)
                    for i, r in matches.iterrows():
                        with cols[i % 3]:
                            name = r["display_name"]
                            st.markdown(
                                f"<div class='product-card'><h4>{name}</h4><p> ₹{(r['price'] or 0):.2f} |  ₹{(r.get('freight_value') or 0):.2f}</p></div>",
                                unsafe_allow_html=True,
                            )
                            if st.button("🛒 Add to Cart", key=f"imgadd_{i}"):
                                st.session_state.cart.append({"product": name, "price": float(r["price"] or 0)})
                                st.success(f"Added {name} to cart.")
                else:
                    st.warning("No similar items found for this image.")


# -------------------------
# ANALYTICS (with insights)
# -------------------------
elif st.session_state.page == "Analytics":
    st.header(" Analytics — Business Insights")
    st.markdown("<div class='analytics-insight'>Ask a question like “Which category generated most revenue?”</div>", unsafe_allow_html=True)
    question = st.text_input("Ask a question:")
    if st.button("Generate & Run SQL"):
        with st.spinner("Generating SQL..."):
            sql = nl_to_sql_with_gemini(question, df.columns.tolist())
        st.code(sql, language="sql")

        if "select" not in sql.lower():
            st.error("Gemini did not return a valid SELECT query.")
        else:
            try:
                con = duckdb.connect(database=":memory:")
                con.register("df", df)
                result = con.execute(sql).df()
                con.close()
            except Exception as e:
                st.error(f"SQL error: {e}")
                result = pd.DataFrame()

            if not result.empty:
                st.dataframe(result)
                st.bar_chart(result.select_dtypes("number"))
                with st.spinner("Generating explanation..."):
                    explain_prompt = f"""
You are a data analyst. Summarize the following SQL output in 3 sentences:
Question: {question}
Result sample:
{result.head(10).to_csv(index=False)}
"""
                    explanation = genai.GenerativeModel("models/gemini-2.5-flash").generate_content(explain_prompt)
                    st.markdown(f"<div class='analytics-insight'>{explanation.text}</div>", unsafe_allow_html=True)

# -------------------------
# Footer
# -------------------------
st.markdown("<div style='text-align:center;color:#bde9ff;margin-top:18px;'>Pluto — GenAI E-commerce Assistant © 2025</div>", unsafe_allow_html=True)