from groq import Groq
from settings import api, models, prompts
import json
from datetime import datetime

Bloom_prompt = prompts.BLOOM_QUESTION_GENERATION_PROMPT

def get_response_from_llm(data_json, prompt):
    client = Groq(api_key=api.GROQ_API_KEY)
    prompt = str(data_json)  + prompt # Get next prompt from the cycle

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=models.MODEL_NAME
    )
    return response.choices[0].message.content


if __name__ == '__main__':
    # Replace with your actual data
    prompt = Bloom_prompt
    json_data = {
        "The topic questions": """ Exercice 1 : Définition et Types d'Ondes

    Qu'est-ce qu'une onde mécanique ? Donnez une définition concise.
    Expliquez la différence entre une onde transversale et une onde longitudinale avec un exemple pour chacune.

Exercice 2 : Propagation et Vitesse

    Comment la vitesse de propagation d'une onde est-elle définie dans un milieu ? Utilisez la formule donnée dans vos notes.
    Quel effet l’élasticité d’un milieu a-t-elle sur la vitesse de propagation d’une onde ? Donnez un exemple.

Exercice 3 : Expérience sur les Ondes Sonores

    Si une onde sonore nécessite un milieu matériel pour se propager, pourquoi le son ne se propage-t-il pas dans le vide ?

Exercice 4 : Superposition d’Ondes

    Qu'est-ce que le principe de superposition pour les ondes mécaniques ? Illustrer par un exemple simple ce que cela signifie lorsque deux ondes se rencontrent.

Exercice 5 : Calcul de la Vitesse et du Retard

    En utilisant la relation V=dΔtV=Δtd​, calculez la célérité d’une onde si la distance parcourue est de 340 mètres et le temps pris est de 1 seconde.
    Si le retard ττ est la différence de temps pour que l'onde atteigne deux points différents dans le milieu, comment calculeriez-vous le retard entre deux points situés à 170 mètres l'un de l'autre, si la vitesse de l'onde est de 340 m/s ?""",
    }
    bloom = get_response_from_llm(json_data, prompt)
    print(bloom)
# Parse the response and save to JSON file with a unique name based on current date and time
    try:
        # Function to extract JSON from within code blocks
        def extract_json_from_story(story_text):
            import re
            # Regex to find JSON within ```json ... ```
            json_pattern = r"```json\s*(\{.*?\})\s*```"
            match = re.search(json_pattern, story_text, re.DOTALL)
            if match:
                return match.group(1)
            # If not found, try to find JSON within ```
            json_pattern_generic = r"```\s*(\{.*?\})\s*```"
            match = re.search(json_pattern_generic, story_text, re.DOTALL)
            if match:
                return match.group(1)
            # If no code blocks, assume entire response is JSON
            return story_text.strip()

        # Extract JSON string
        json_str = extract_json_from_story(bloom)

        # Parse the JSON string
        generated_json = json.loads(json_str)

        # Get current date and time for unique filename
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_taxonomy_{current_datetime}.json"

        # Save the JSON to the file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(generated_json, f, ensure_ascii=False, indent=2)

        print(f"Generated taxonomy questions have been saved to '{filename}'.")
    except json.JSONDecodeError:
        print("Failed to parse the LLM response as JSON. Please ensure the response is in valid JSON format.")
