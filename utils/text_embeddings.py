import os

from openai import OpenAI

def prompt_propcess(text):
    num = ['1st', '2nd', '3rd']
    prompt_text = ''
    c = 0
    s = ''
    for ch in text:
        if ch == '|':
            if c == 0:
                prompt_text += 'Most importantly, the 1st diagnosis is {' + s + '}.'
            else:
                prompt_text += 'As a supplementary condition, the ' + (
                    num[c] if c <= 2 else str(c + 1) + 'th'
                ) + ' diagnosis is {' + s + '}.'
            c += 1
            s = ''
        else:
            s += ch
    if s != '':
        if c == 0:
            prompt_text += 'Most importantly, the 1st diagnosis is {' + s + '}.'
        else:
            prompt_text += 'As a supplementary condition, the ' + (
                num[c] if c <= 2 else str(c + 1) + 'th'
            ) + ' diagnosis is {' + s + '}.'
    return prompt_text


def _get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it before running generation or evaluation "
            "that requires text embeddings."
        )
    return OpenAI(api_key=api_key)


def get_text_embedding(text):
    prompt_text = prompt_propcess(text)

    response = _get_openai_client().embeddings.create(
        model="text-embedding-3-small",
        input=[prompt_text]
    )
    return response.data[0].embedding


def get_text_embedding_v0(text):
    return get_text_embedding(text)


def get_text_embedding_v1(text):
    return get_text_embedding(text)
