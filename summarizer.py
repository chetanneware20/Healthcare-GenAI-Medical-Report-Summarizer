from openai import OpenAI

client = OpenAI()

def summarize_report(text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a medical report summarizer."},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content
