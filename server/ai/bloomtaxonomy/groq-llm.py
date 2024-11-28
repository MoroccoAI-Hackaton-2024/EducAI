from groq import Groq
from settings import api, models, prompts

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
    story = get_response_from_llm(json_data, prompt)
    print(story)