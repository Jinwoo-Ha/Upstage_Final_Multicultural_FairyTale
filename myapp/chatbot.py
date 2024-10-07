import json
from typing import List, Dict, Any, Optional
from pprint import pprint
import os
import re

from django.conf import settings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_upstage import ChatUpstage
from langchain_community.tools import DuckDuckGoSearchResults
from openai import OpenAI

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=settings.UPSTAGE_API_KEY, base_url="https://api.upstage.ai/v1/solar")

# ChatUpstage 모델 초기화
solar_mini = ChatUpstage(model="solar-1-mini-chat")

def extracted_claimed_facts(text: str, llm: ChatUpstage = solar_mini) -> list:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert fact extractor. Extract claimed facts from the given text, focusing on entities and their relationships."),
        ("human", "Extract the claimed facts from the following text, providing a list of dictionaries. Each dictionary should represent a fact and include keys for 'entity', 'relation', and 'value'. Be specific and precise with the relations.\n\n{input_text}"),
        ("human", "Respond with a JSON array of fact dictionaries only, without any additional text.")
    ])

    output_parser = JsonOutputParser()
    chain = prompt | llm | output_parser

    result = chain.invoke({"input_text": text})
    return result

def search_context(text: str, claimed_facts: list, search_tool: DuckDuckGoSearchResults = DuckDuckGoSearchResults(), llm: ChatUpstage = solar_mini) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at generating concise and relevant search keywords. Generate a list of 3-5 search keywords or short phrases based on the given text and extracted facts."),
        ("human", "Given the following text and extracted facts, generate a list of 3-5 search keywords or short phrases:\n\nText: {text}\n\nExtracted Facts:\n{facts}\n\nProvide only the keywords or short phrases, separated by commas.")
    ])

    facts_str = "\n".join([f"- {fact['entity']} {fact['relation']} {fact['value']}" for fact in claimed_facts])
    keywords_response = llm.invoke(prompt.format(text=text, facts=facts_str))

    keywords = [kw.strip() for kw in keywords_response.content.split(",") if kw.strip()]
    search_query = " ".join(keywords)
    search_results = search_tool.run(search_query)

    return search_results

def build_kg(claimed_facts: list, context: str, llm: ChatUpstage = solar_mini) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert in building knowledge graphs. Construct a knowledge graph based on the given context, using the claimed facts only as inspiration for the schema."),
        ("human", "Given the following context and claimed facts, construct a knowledge graph. Assume all information in the context is true, but use the claimed facts only as hints for the types of relations to look for.\n\nContext:\n{context}\n\nClaimed Facts (use only as schema hints):\n{claimed_facts}\n\nConstruct the knowledge graph as a JSON object where keys are entities and values are dictionaries of relations. Each relation should have a \"value\" and a \"source\" (a relevant quote from the context).\n\nConstruct the knowledge graph:")
    ])

    output_parser = JsonOutputParser()
    chain = prompt | llm | output_parser

    facts_str = "\n".join([f"- {fact['entity']} {fact['relation']} {fact['value']}" for fact in claimed_facts])
    kg = chain.invoke({"context": context, "claimed_facts": facts_str})

    return kg

def verify_facts(claimed_facts: list, context: str, kg: dict, confidence_threshold: float, llm: ChatUpstage = solar_mini) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert fact-checker. Verify a claimed fact against a knowledge graph and context information."),
        ("human", "Verify the following claimed fact using the provided knowledge graph and context. Determine if it's verified, assign a confidence score (0.0 to 1.0), and provide a brief explanation.\n\nClaimed Fact: {entity} {relation} {value}\n\nKnowledge Graph:\n{kg}\n\nContext:\n{context}\n\nProvide a JSON object with the following structure:\n{{\n  \"verified\": str,  # Must be either \"TRUE\" or \"FALSE\"\n  \"confidence\": float,\n  \"explanation\": string\n}}\n\nProvide the verification result:")
    ])

    output_parser = JsonOutputParser()
    chain = prompt | llm | output_parser

    kg_str = json.dumps(kg, indent=2)
    verified_facts = {}

    for i, fact in enumerate(claimed_facts):
        verification_result = chain.invoke({
            "entity": fact["entity"],
            "relation": fact["relation"],
            "value": fact["value"],
            "kg": kg_str,
            "context": context,
        })

        verification_result["verified"] = "TRUE" if verification_result["verified"] else "FALSE"

        verified_facts[str(i)] = {
            "claimed": f"{fact['entity']} {fact['relation']} {fact['value']}",
            **verification_result,
        }

    return verified_facts

def add_fact_check_comments_to_text(text, verified_facts, llm=solar_mini):
    fact_map = {fact["claimed"]: fact for fact in verified_facts.values()}

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant tasked with adding fact-check annotations to a given text. For each fact in the text that has been verified, add an inline annotation right after the fact, using the following format: [Fact: <STATUS> (Confidence: <CONFIDENCE>) - <BRIEF_EXPLANATION>] Where <STATUS> is True, False, or Unsure, <CONFIDENCE> is the confidence score, and <BRIEF_EXPLANATION> is a very short explanation."),
        ("human", "Original text:\n{text}\n\nVerified facts:\n{fact_map}\n\nPlease add fact-check annotations to the original text based on the verified facts.")
    ])

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"text": text, "fact_map": fact_map})

    return response

def fact_check(text_to_check):
    claimed_facts = extracted_claimed_facts(text_to_check)
    relevant_context = search_context(text_to_check, claimed_facts)
    kg = build_kg(claimed_facts, relevant_context)

    verified_facts = verify_facts(claimed_facts, relevant_context, kg, confidence_threshold=0.7)

    fact_checked_text = add_fact_check_comments_to_text(text_to_check, verified_facts)
    
    return fact_checked_text

def perc_fact(text):
    fact_patterns = re.findall(r'\[Fact: (True|False).+?\]', text)
    
    success = 0

    if len(fact_patterns) != 0:
        for fact in fact_patterns:
            if "True" in fact:
                success += 1
    
        return success / len(fact_patterns) * 100
    else:
        return 100  # 인식된 사실이 없을 경우 일상적인 대화로 간주

def classify_response(answer):
    classification_prompt = f"""
    You are a classifier. Classify the following assistant response into one of two categories: 
    1) Informative: The response contains factual or educational information about a topic. 
    2) Casual: The response is a greeting or casual conversation that doesn't need fact-checking. 
    
    Assistant response: "{answer}"
    """

    response = client.chat.completions.create(
        model="solar-1-mini-chat",
        temperature=0.3,
        messages=[{"role": "user", "content": classification_prompt}]
    )
    classification_result = response.choices[0].message.content.strip().lower()

    return classification_result

def ask_chatbot(user_input):
    messages = [
        {
            "role": "system", 
            "content": """ 너의 이름은 '호솔'이야.
            너의 역할은 동화를 읽고 있는 어린이의 이해를 돕기 위해 질문에 답하는 것이야.
            너는 친근한 반말로 한국에 사는 어린이가 다문화적인 관점을 기를 수 있도록 도와야 해.

            모든 문장은 반말이어야 하고, 질문에 대해 가장 정확한 답변을 내야 해.
            어린이들이 이해하기 쉽게 간단한 단어와 문장 구조를 사용해.
            어린이가 어려워하거나 힘들어하면 격려해줘야 해.

            반드시 모든 문장을 반말로 해야 하고, 70자 이내로 답변해줘.

            Your answer must always be written in three sentences.
            """
        },
        # few-shot examples
        {"role": "user", "content": "안녕? 넌 누구야?"},
        {"role": "assistant", "content": "안녕? 나는 호솔이라고 해. 무엇이 궁금하니?"},

        {"role": "user", "content": "일본에 대해서 알려줘!"},
        {"role": "assistant", "content": "일본은 벚꽃이 정말 예쁘고 후지산이라는 높은 산이 있어. 일본 사람들은 스시라는 음식을 많이 먹는데, 생선과 밥으로 만든 특별한 음식이야."},

        {"role": "user", "content": "사막은 어떻게 생겼어?"},
        {"role": "assistant", "content": "사막은 아주 넓고 뜨거운 모래로 덮인 곳이야. 낮에는 아주 뜨겁고, 밤에는 추워져. 하지만 가끔 선인장이나 작은 동물들도 볼 수 있어."},
        
        {"role": "user", "content": "기린은 왜 목이 길어?"},
        {"role": "assistant", "content": "기린은 나무 높은 곳에 있는 잎을 먹기 위해 목이 길어! 목이 길어서 멀리서도 다른 기린이나 포식자를 잘 볼 수 있어."},
    ]

    messages.append({"role": "user", "content": user_input})
    
    chat_result = client.chat.completions.create(
        model="solar-1-mini-chat",
        messages=messages,
        temperature=0.3
    )
    
    assistant_response = chat_result.choices[0].message.content

    if "informative" in classify_response(assistant_response):
        try: 
            fact_checked_response = fact_check(assistant_response)
            fact_checked_output = perc_fact(fact_checked_response)

            if fact_checked_output < 50:
                assistant_response = chat_result.choices[0].message.content
                fact_checked_response = fact_check(assistant_response)
                fact_checked_output = perc_fact(fact_checked_response)
        except Exception as e:
            return "미안, 혹시 다시 한번 말해줄 수 있어?"

    return assistant_response