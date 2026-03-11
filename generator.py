def generate_answer(contexts, question):
    from groq import Groq
    import os

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    MAX_TOTAL_CHARS = 4000  # safe total context budget
    accumulated_context = ""
    
    for ctx in contexts:
        if len(accumulated_context) + len(ctx) > MAX_TOTAL_CHARS:
            break
        accumulated_context += ctx + "\n\n"

    prompt = f"""
You are a helpful assistant.
Answer the question using ONLY the given context.
If the answer is not present, say you cannot find it.

Context:
{accumulated_context}

Question:
{question}

Answer:
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"LLM Error: {str(e)}"