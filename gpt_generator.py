import os
import requests
import base64
import pandas as pd
import pickle
import json
import random
import ast
import time

# Configuration
GPT4V_KEY = "YOUR_TOKEN_HERE"

headers = {
    "Content-Type": "application/json",
    "api-key": GPT4V_KEY,
}

# Endpoint for GPT-4 API
GPT4V_ENDPOINT = "https://lakmoosgpt.openai.azure.com/openai/deployments/gpt4-turbo/chat/completions?api-version=2024-08-01-preview"


conversation_history = []


def delete_history():
    global conversation_history
    conversation_history = []


def log_history_length():
    print("history length: ", len(conversation_history))
    # print history size in mb
    print(
        "history size: ",
        round(len(pickle.dumps(conversation_history)) / 1024, 2),
        " kb",
    )

    # if history is longer than 10 messages, delete the first two messages
    if len(conversation_history) > 8:
        # remove first two messages
        conversation_history.pop(0)
        conversation_history.pop(0)


def send_prompt(role, content):
    global conversation_history
    log_history_length()

    # Add the new message to the conversation history

    # add content of to conversation history
    conversation_history.append({"role": role, "content": content})

    # Payload for the request
    payload = {
        "messages": conversation_history,
        "temperature": 0.7,
        "max_tokens": 1800,
    }

    try:
        response = requests.post(GPT4V_ENDPOINT, headers=headers, json=payload)
        response_data = response.json()
        # Get the assistant's reply and add it to the conversation history
        assistant_message = response_data["choices"][0]["message"]["content"]
        conversation_history.append({"role": "assistant", "content": assistant_message})

        return assistant_message
    except Exception as e:
        # print response status code and error message
        print(response.status_code)

        print(e)
        return None


# Define the prompt for generating personas
generate_persona_prompt = """
    Vygeneruj profil náhodné osoby z České republiky: 
    Vygeneruj parametry které jsou v následujícím ukázkovém jsonu, 
    Distribuce všech parametrů by měla reprezentovat rozložení v české populaci.
    Distribuce měst, by měla reflektovat počet obyvatel v jednotlivých městech.
    Volební preference by měly být pro únor 2024.
    Ukázka výstupu. Vrať jen následující validní JSON objekt: 
    {{
        "name": "jméno",
        "age": "věk",
        "gender": "pohlaví",
        "residence": "bydliště",
        "income": "czk/měsíc",
        "education_level": "vzdělání",
        "yearly_energy_consumption_kWh": "roční spotřeba energie v kWh",
        "political_preferences": "Preferovaná politická strana"
    }}
"""


# Send request and get response

# create dataframe for personas
df = pd.DataFrame(
    columns=[
        "name",
        "age",
        "gender",
        "residence",
        "income",
        "education_level",
        "yearly_energy_consumption_kWh",
        "political_preferences",
    ]
)


delete_history()

personas = 1000

# try to load response data from pickle file
for i in range(personas):
    # sleep for a random time between 1 and 2 seconds float
    time.sleep(random.uniform(1, 2))

    data = send_prompt(
        "user",
        generate_persona_prompt,
    )

    data = data.replace("```json", "").replace("```", "")

    data = json.loads(data)

    print(data)

    # iterate over column names and add data to dataframe

    for column_name in df.columns:
        df.loc[i, column_name] = data[column_name]

    # create savepoint for every 10 personas
    if i % 10 == 0:
        df.to_csv(f"personas.csv", index=False)
        print(f"Saved {i} personas to file.")

df.to_csv(f"new_personas.csv", index=False)
