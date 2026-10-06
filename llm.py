from typing import List, Dict, Generator
from openai import OpenAI

OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL_NAME = "qwen2.5-coder:1.5b"

class LocalRAGLLM:
    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = MODEL_NAME):
        self.model = model
        self.client = OpenAI(
            base_url=base_url,
            api_key="ollama"
        )

    def stream_answer(self, query: str, context_chunks: List[Dict]) -> Generator[str, None, None]:
        context_blocks = []
        for c in context_chunks:
            citation_tag = f"[{c['filename']}, Page {c['page_number']}]"
            context_blocks.append(f"Source {citation_tag}:\n{c['text_content']}")

        context_str = "\n\n---\n\n".join(context_blocks)

        system_prompt = (
            "You are an Intelligent University Knowledge Assistant. "
            "Your task is to answer user queries strictly using the provided context.\n\n"
            "Rules:\n"
            "1. Rely ONLY on clear facts mentioned directly in the context below.\n"
            "2. Do not hallucinate, generalize, or import outside institutional knowledge.\n"
            "3. If the context does not contain sufficient facts to answer the question, state: "
            "\"I do not have enough information in the provided context to answer this question.\"\n"
            "4. For every claim, cite the exact source using the format: [DocumentName, Page X].\n"
            "5. Keep the tone academic, precise, and concise."
        )

        user_content = f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.0,
            stream=True
        )

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
