import os
import re
import torch
import json
import asyncio
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Model Import
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.empty_cache()

print("[LangChain] Torch CUDA Available : ", torch.cuda.is_available())
print("[LangChain] Current Device : ", device)

model_id = "llama3"

print("[LangChain] Importing LLM Model :", model_id)
llm = ChatOllama(model=model_id, device=device)
print("[LangChain]-[" + model_id + "]", llm.invoke("Hello World!"))
print("[LangChain] Imported LLM Model :", model_id)

def parse_JSON(llm_response, is_array=False):
    json_pattern = re.compile(r'{[^{}]*?}')

    # LLM 응답에서 JSON 값 찾기
    json_match = json_pattern.findall(llm_response)
    
    if json_match and is_array:
        json_array = []
        for string in json_match:
            try:
                json_array.append(json.loads(string))
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {str(e)}")
                print(string)
        return json_array
    elif json_match:
        json_str = json_match[-1]
        try:
            json_data = json.loads(json_str)
            return json_data
        except json.JSONDecodeError:
            print("[LangChain]-[parse_JSON] Invalid JSON format")
            return None
    else:
        print("[LangChain]-[parse_JSON] No JSON found in the LLM response")
        return None
    
######## Prompt LangChain ########
async def generate_problems(script, subject, problem_num, problem_types):
    print("\n[LangChain]-[generate_problems] Subject :", subject)
    print("[LangChain]-[generate_problems] Problem_num :", problem_num)
    print("[LangChain]-[generate_problems] Problem Types : ", problem_types, "\n")

    prompt = ChatPromptTemplate.from_template("""
        당신은 대한민국 대학교 {subject} 교수입니다.
        당신은 학생들의 학습 수준을 평가하기 위해서 시험 문제를 출제하는 중입니다.

        {script}

        위 스크립트는 대한민국의 대학교 수준의 {subject}강의 내용인데
        이때 위 스크립트에 기반하여 {problem_num} 개의 문제를 JSON 형식으로 아래 조건에 맞추어서 생성해주세요.

        1. 문제의 Type은 아래와 같이 총 4개만 존재합니다.

        MultipleChoice : 객관식, Option은 네개, 즉 사지선다형
        ShrotAnswer : 단답형
        BlanckQuestion : 빈칸 뚫기 문제
        OXChoice : O X 문제

        2. 주어진 스크립트에서 시험에 나올 수 있는, 중요한 부분에 대한 문제를 생성해주세요.
        3. 추가적인 설명 없이 JSON 결과만 제공해주세요.
        4. 문제 JSON은 아래와 같은 형태여야만 합니다.

            [
                {{
                    "type": "",
                    "direction": "",
                    "options": [
                    "",
                    "",
                    "",
                    ""
                    ],
                    "answer": ""
                }},
                {{
                    // 다음 문제
                }},
                ...
            ]

        아래는 각 JSON의 요소들에 대한 설명입니다. 아래의 설명에 완벽하게 맞추어서 생성해주세요.

        type : 문제 Type 4개 중에 1개

        direction : 문제 질문
        direction : type이 BlanckQuestion인 경우에는 direction에 ___로 빈칸을 뚫어야 한다
        direction : type이 OXChoice인 경우에는 direction이 질문 형태가 아닌 서술 형태로 참 또는 거짓일 수 있어야 한다

        options: MultipleChoice인 경우에만 보기 4개
        options: MultipleChoice이 아닌 다른 Type이면 빈 배열
        options : OXChoice인 경우에도 빈 배열

        answer : 각 문제들에 대한 정답
        answer : MultipleChoice인 경우 options들 중 정답 번호
        answer : ShrotAnswer의 경우 direction에 대한 정답
        answer : BlanckQuestion인 경우 direction에 뚫린 빈칸
        answer : OXChoice인 경우 X인 경우 answer는 0, O인 경우 answer는 1

        5. 이 중에서 {problem_types}에 해당하는 종류의 문제만 생성해주세요
        6. 각 문제의 Type에 맞는 JSON 요소들을 생성해주세요
        7. 항상 모든 문제에 대한 direction과 answer는 꼭 생성해주세요
        8. 문제는 모두 한국어로 생성해주세요
        9. 이를 생성할 때 고민의 시간을 가지고 정확하게 생성해주새요
    """)

    chain = (
        prompt 
        | llm 
        | StrOutputParser()
    )

    problem_result = await asyncio.to_thread(
        chain.invoke, {
               "script" : script,
               "subject" : subject,
               "problem_num" : problem_num,
               "problem_types" : problem_types
        })

    json_result = parse_JSON(problem_result, True)

    if not json_result:
        return None
    
    return json_result