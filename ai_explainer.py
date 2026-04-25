# ai_explainer.py

import os
from dotenv import load_dotenv
from groq import Groq

# This reads the key from .env file
# so your real key is never exposed
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=API_KEY)


def explain_risk(stock_name, investment_amount, time_period,
                 loss_probability, avg_final_value,
                 annual_volatility, mean_daily_return):
    """
    Sends simulation results to Groq LLM.
    Returns plain language explanation.
    """

    avg_gain_loss  = avg_final_value - investment_amount
    gain_loss_word = "gain" if avg_gain_loss > 0 else "loss"

    prompt = f"""
    You are a friendly financial advisor explaining investment risk
    to a young Indian investor (age 18-25) who is new to investing.
    Use very simple language. No jargon. Be honest but encouraging.
    Keep your explanation under 150 words.

    Here is the data from our simulation:
    - Stock: {stock_name}
    - Amount invested: ₹{investment_amount:,}
    - Time period: {time_period}
    - Loss probability: {loss_probability:.1f}%
    - Average expected final value: ₹{avg_final_value:,.0f}
    - Average expected {gain_loss_word}: ₹{abs(avg_gain_loss):,.0f}
    - Annual volatility: {annual_volatility:.1f}%
    - Average daily return: {mean_daily_return * 100:.4f}%

    Please explain:
    1. What the loss probability means in simple words
    2. Whether this stock looks good or risky right now
    3. One practical tip for this investor

    Talk directly to the investor using "you" and "your money".
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role"   : "system",
                "content": "You are a helpful and friendly financial advisor for young Indian investors."
            },
            {
                "role"   : "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content