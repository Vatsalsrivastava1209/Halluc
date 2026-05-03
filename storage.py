import sqlite3
import logging
import os

logger = logging.getLogger(__name__)
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tracker_v2.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS evaluations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT, category TEXT, topic TEXT, requested_model TEXT, exact_model TEXT,
        factuality_score INTEGER, hallucination_detected INTEGER,
        overconfidence_score REAL, evasiveness_score INTEGER,
        bias_detected INTEGER, bias_type TEXT, bias_explanation TEXT,
        explanation TEXT, raw_answer TEXT
    )""")
    conn.commit()
    return conn

def save_evaluation(daily_data: dict, model_answers: dict, evaluation_results: dict):
    conn = init_db()
    for model, data in evaluation_results.items():
        exact_model = data["exact_model"]
        score = data["score"]
        raw_answer = model_answers.get(model, "")
        
        conn.execute("""INSERT INTO evaluations VALUES 
            (NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
            daily_data["date"], daily_data["category"], daily_data["topic"], model, exact_model,
            score.factuality_score, int(score.hallucination_detected),
            score.overconfidence_score, score.evasiveness_score,
            int(score.bias_detected), score.bias_type, score.bias_explanation,
            score.explanation, raw_answer
        ))
    conn.commit()
    logger.info("💾 Saved to tracker_v2.db")