from flask import Flask, request, render_template
import psycopg2
from sentence_transformers import SentenceTransformer, util
import json
from jinja2 import Template
import psycopg2.extras
import torch
import numpy as np
import requests
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
model = SentenceTransformer('all-MiniLM-L6-v2')
user_sessions = {}

API_KEY = os.getenv('API_KEY')
DB_NAME = os.getenv('DB_NAME')
DB_PORT = os.getenv('DB_PORT')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')

CONFIG_TEMPLATE = Template("""
strategy: {{ strategy }}
legs:
{% for leg in legs %}
  - type: {{ leg.type }}
    strike: {{ leg.strike }}
    action: {{ leg.action }}
{% endfor %}
risk_profile: {{ risk_profile }}
""")


def get_db_connection():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )


def insert_db():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS strategies (
            name TEXT PRIMARY KEY,
            data JSONB,
            embedding BYTEA
        )
    ''')

    strategies = {
        "long strangle": {
            "strategy": "long_strangle",
            "legs": [
                {"type": "call", "strike": 18000, "action": "buy"},
                {"type": "put", "strike": 17500, "action": "buy"}
            ],
            "risk_profile": "limited_loss_unlimited_gain"
        },
        "bull call spread": {
            "strategy": "bull_call_spread",
            "legs": [
                {"type": "call", "strike": 17500, "action": "buy"},
                {"type": "call", "strike": 18000, "action": "sell"}
            ],
            "risk_profile": "limited_loss_limited_gain"
        }
    }

    for name, data in strategies.items():
        embedding = model.encode(name).astype('float32').tobytes()
        cur.execute("""
        INSERT INTO strategies (name, data, embedding)
        VALUES (%s, %s, %s)
        ON CONFLICT (name) DO UPDATE SET data = EXCLUDED.data, embedding = EXCLUDED.embedding
        """, (name, json.dumps(data), embedding))

    conn.commit()
    cur.close()
    conn.close()


def get_strategy(user_input):
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute('SELECT name, embedding FROM strategies')
    rows = cur.fetchall()

    input_vec = model.encode(user_input, convert_to_tensor=True)
    similarities = []
    for row in rows:
        name = row['name']
        emb_blob = row['embedding']
        emb_array = np.frombuffer(emb_blob, dtype=np.float32)
        emb_tensor = torch.from_numpy(emb_array)
        sim = util.cos_sim(input_vec, emb_tensor).item()
        similarities.append((sim, name))

    cur.close()
    conn.close()
    if not similarities:
        return None
    best_match = max(similarities, key=lambda x: x[0])
    return best_match[1] if best_match[0] > 0.7 else None


def query_groq_api(question, context=None):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    messages = []
    if context:
        messages.append({
            "role": "system",
            "content": f"You are a financial assistant. You have to give right and accurate advice. "
                       f"Use the website https://optionstrat.com/ to know the option strategy format. Use the following context:\n{context}"
        })
    messages.append({
        "role": "user",
        "content": question
    })
    payload = {
        "model": "llama3-8b-8192",
        "messages": messages,
        "max_tokens": 500
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        return f"Error contacting Groq API: {str(e)}"
    except (KeyError, IndexError):
        return "Unexpected response format from Groq API."


def generate_strategy_config(strategy_name):
    prompt = f"""
    Generate a trading strategy configuration for '{strategy_name}' in the following YAML format:
    strategy: <strategy_name>
    legs:
      - type: <call or put>
        lot: <lot size>
        strike: <strike price as integer>
        action: <buy or sell>
      - type: <call or put>
        lot: <lot size>
        strike: <strike price as integer>
        action: <buy or sell>
    risk_profile: <limited_loss_unlimited_gain, limited_loss_limited_gain, unlimited_loss_limited_gain, unlimited_loss_unlimited_gain etc.>
    Ensure the configuration is realistic and follows typical options trading principles for {strategy_name}.
    Return only the YAML content, nothing else.
    """
    response = query_groq_api(prompt)
    if "Error" in response or "Unexpected" in response:
        return None, response

    # Parse the YAML-like response into JSON for database storage
    try:
        lines = response.strip().split("\n")
        strategy_data = {"strategy": strategy_name, "legs": [], "risk_profile": ""}
        current_section = None
        for line in lines:
            line = line.strip()
            if line.startswith("strategy:"):
                strategy_data["strategy"] = line.split(":")[1].strip()
            elif line.startswith("legs:"):
                current_section = "legs"
            elif line.startswith("risk_profile:"):
                strategy_data["risk_profile"] = line.split(":")[1].strip()
                current_section = None
            elif current_section == "legs" and line.startswith("- type:"):
                leg = {
                    "type": line.split(":")[1].strip(),
                    "strike": 0,
                    "action": ""
                }
                strategy_data["legs"].append(leg)
            elif current_section == "legs" and line.startswith("strike:"):
                strategy_data["legs"][-1]["strike"] = int(line.split(":")[1].strip())
            elif current_section == "legs" and line.startswith("action:"):
                strategy_data["legs"][-1]["action"] = line.split(":")[1].strip()

        # Validate the data
        if not strategy_data["legs"] or not strategy_data["risk_profile"]:
            return None, "Invalid strategy configuration generated."

        # Store in database
        conn = get_db_connection()
        cur = conn.cursor()
        embedding = model.encode(strategy_name).astype('float32').tobytes()
        cur.execute("""
        INSERT INTO strategies (name, data, embedding)
        VALUES (%s, %s, %s)
        ON CONFLICT (name) DO UPDATE SET data = EXCLUDED.data, embedding = EXCLUDED.embedding
        """, (strategy_name, json.dumps(strategy_data), embedding))
        conn.commit()
        cur.close()
        conn.close()

        # Render the config for display
        config_output = CONFIG_TEMPLATE.render(**strategy_data)
        return config_output, None
    except Exception as e:
        return None, f"Failed to parse strategy configuration: {str(e)}"


@app.route("/", methods=["GET", "POST"])
def index():
    reply = ""
    last_config = ""
    if request.method == "POST":
        user_id = "1"
        message = request.form.get("message", "").strip()
        session = user_sessions.get(user_id, {})
        last_config = session.get("last_config", "")

        if any(word in message.lower() for word in ["setup", "generate", "strategy", "show", "config"]):
            strategy_name = get_strategy(message)
            if strategy_name:
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute("SELECT data FROM strategies WHERE name = %s", (strategy_name,))
                row = cur.fetchone()
                cur.close()
                conn.close()

                data = row[0]
                config_output = CONFIG_TEMPLATE.render(**data)
                user_sessions[user_id] = {"last_config": config_output}
                reply = f"Here is the config for {strategy_name} strategy:\n\n{config_output}"
                last_config = config_output
            else:
                # Strategy not found, generate a new one
                strategy_name = message.lower().replace("generate", "").replace("show", "").replace("setup","").replace(
                    "strategy", "").replace("config", "").strip()
                if strategy_name:
                    config_output, error = generate_strategy_config(strategy_name)
                    if error:
                        reply = f"Could not generate strategy: {error}"
                    else:
                        user_sessions[user_id] = {"last_config": config_output}
                        reply = f"Generated new config for {strategy_name} strategy:\n\n{config_output}"
                        last_config = config_output
                else:
                    reply = "Please specify a strategy name."
        elif last_config:
            followup_reply = query_groq_api(message, last_config)
            reply = followup_reply
        else:
            reply = "Hi! Tell me a strategy name or what you’d like to do."

    return render_template("index.html", reply=reply)


if __name__ == "__main__":
    insert_db()
    app.run(debug=True)