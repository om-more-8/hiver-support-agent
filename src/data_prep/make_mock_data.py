"""
Generates a small mock CSV matching the REAL schema of the Kaggle
"Customer Support on Twitter" dataset (thoughtvector/customer-support-on-twitter):

    tweet_id, author_id, inbound, created_at, text,
    response_tweet_id, in_reply_to_status_id

This lets you build and test the entire pipeline right now, before
downloading the real ~3M-row dataset from Kaggle.

USAGE:
    python src/data_prep/make_mock_data.py
Writes to: data/raw/twcs.csv

Once you have the real file, just drop it at that same path
(same filename, same columns) and every downstream script works unchanged.
"""
import pandas as pd
import random
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import RAW_CSV, BRAND

random.seed(42)

# Realistic-ish complaint/resolution templates across a handful of intents.
# In the real dataset these patterns exist naturally across thousands of threads.
TEMPLATES = [
    {
        "intent": "delivery_delay",
        "customer": "My order #{oid} still hasn't arrived, it's been {days} days past the estimate. Where is it??",
        "brand_reply": "Sorry for the wait! I've checked order #{oid} and it's currently in transit, expected within 2 days. I'll send you tracking details via DM.",
    },
    {
        "intent": "refund_request",
        "customer": "I returned item #{oid} two weeks ago and still no refund. This is ridiculous.",
        "brand_reply": "I'm sorry about the delay on your refund for #{oid}. I've escalated this to our refunds team, you should see it within 3-5 business days.",
    },
    {
        "intent": "account_login",
        "customer": "Can't log into my account, it keeps saying invalid password even after reset. Help!",
        "brand_reply": "Sorry for the trouble! Please try clearing your browser cache and resetting again via the link we just sent. Let us know if it persists.",
    },
    {
        "intent": "damaged_item",
        "customer": "Order #{oid} arrived completely broken. Packaging was fine but the item inside is smashed.",
        "brand_reply": "So sorry to hear that! I've started a replacement for order #{oid}, no need to return the damaged item. It'll ship within 24 hours.",
    },
    {
        "intent": "billing_dispute",
        "customer": "I was charged twice for order #{oid}, please fix this immediately.",
        "brand_reply": "Apologies for the duplicate charge on #{oid}. I've flagged this with billing and the extra charge will be reversed within 2 business days.",
    },
    {
        "intent": "general_inquiry",
        "customer": "Does order #{oid} include international shipping or is that separate?",
        "brand_reply": "Great question! International shipping is included in the price shown at checkout for order #{oid}, no extra charge.",
    },
    {
        "intent": "escalation_worthy_complaint",
        "customer": "This is the THIRD time I've contacted support about order #{oid} with no resolution. I want a manager to call me NOW or I'm disputing the charge with my bank.",
        "brand_reply": "I completely understand your frustration and I'm escalating this to a senior specialist right now, they'll reach out directly within the hour.",
    },
]

def generate_rows(n_threads=400):
    rows = []
    tweet_id = 1
    for i in range(n_threads):
        tpl = random.choice(TEMPLATES)
        oid = random.randint(10000, 99999)
        days = random.randint(3, 15)
        cust_text = tpl["customer"].format(oid=oid, days=days)
        brand_text = tpl["brand_reply"].format(oid=oid, days=days)

        cust_id = tweet_id
        brand_id = tweet_id + 1

        rows.append({
            "tweet_id": cust_id,
            "author_id": f"cust_{1000+i}",
            "inbound": True,
            "created_at": "Mon Jan 01 12:00:00 +0000 2024",
            "text": cust_text,
            "response_tweet_id": str(brand_id),
            "in_reply_to_status_id": "",
        })
        rows.append({
            "tweet_id": brand_id,
            "author_id": BRAND,
            "inbound": False,
            "created_at": "Mon Jan 01 12:05:00 +0000 2024",
            "text": brand_text,
            "response_tweet_id": "",
            "in_reply_to_status_id": str(cust_id),
        })
        tweet_id += 2
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = generate_rows()
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_CSV, index=False)
    print(f"Wrote {len(df)} mock rows ({len(df)//2} threads) to {RAW_CSV}")
    print(f"Brand used: {BRAND}")
    print("Replace this file with the real Kaggle CSV (same filename/columns) when ready.")
