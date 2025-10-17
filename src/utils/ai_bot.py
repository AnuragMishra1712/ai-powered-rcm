import requests
import json
import datetime
import os

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def ollama_generate(prompt, model="mistral", log_file="ai_general.log"):
    """
    Sends a prompt to the local Ollama API (Mistral model) and returns AI-generated text.
    Falls back to a simulated response if Ollama isn't running.
    """
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt}

    log_path = os.path.join(LOG_DIR, log_file)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        response = requests.post(url, json=payload, stream=True, timeout=10)
        ai_text = ""
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                ai_text += chunk.get("response", "")
        result = ai_text.strip() or "No response from AI model."
    except Exception as e:
        result = f"[Simulated Response] AI unavailable — simulated output for prompt: {prompt}"

    with open(log_path, "a") as f:
        f.write(f"\n[{timestamp}] PROMPT: {prompt}\nAI RESPONSE: {result}\n{'-'*50}\n")

    return result


# --- Predefined simple AI Bot helpers ---

def pa_followup_bot(claim_id, payer_name, status):
    """Simulates a simple AI follow-up conversation for Prior Authorization."""
    prompt = f"Claim {claim_id} with {payer_name} has PA status '{status}'. Suggest next action."
    return ollama_generate(prompt, log_file="ai_pa_bot.log")


def billing_followup_bot(patient_id, risk_score, balance_due):
    """Simulates a simple AI follow-up conversation for patient billing."""
    prompt = f"Patient {patient_id} has balance ${balance_due} and risk score {risk_score}. Suggest follow-up action."
    return ollama_generate(prompt, log_file="ai_billing_bot.log")

