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

model_id = "llama3.1"

print("[LangChain] Importing LLM Model :", model_id)
llm = ChatOllama(model=model_id, device=device)
print("[LangChain]-[" + model_id + "]", llm.invoke("Hello World!"))
print("[LangChain] Imported LLM Model :", model_id)


def extract_and_parse_json(text):
    # JSON 객체를 찾는 정규 표현식 패턴
    json_pattern = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}')
    
    # 텍스트에서 모든 JSON 객체 찾기
    json_matches = json_pattern.findall(text)
    
    result = []
    for json_str in json_matches:
        try:
            # 중괄호 균형 맞추기
            open_braces = json_str.count('{')
            close_braces = json_str.count('}')
            if open_braces > close_braces:
                json_str += '}' * (open_braces - close_braces)
            
            # 줄바꿈 문자를 이스케이프 처리
            json_str = json_str.replace('\n', '').replace('\r', '')

            # 시작과 끝의 대괄호 제거 (있는 경우)
            json_str = json_str.strip()
            if json_str.startswith('['):
                json_str = json_str[1:]
            if json_str.endswith(']'):
                json_str = json_str[:-1]
            
            # JSON 파싱
            parsed_json = json.loads(json_str)
            
            print(parsed_json)

            result.append(parsed_json)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {str(e)}")
            print(f"Problematic JSON string: {json_str}")

    if not result:
        print("No valid JSON data found in the input text")
        return None

    return result

    
######## Filter LangChain ########
async def filter_office(officeData):
    print("\n[LangChain]-[filter_office] officeData :", officeData)

    prompt = ChatPromptTemplate.from_template("""
        너는 현재 기업들의 워케이션을 위한 데이터를 분류하는 담당자야
        {{
            "name" : "오피스의 이름"
            "description" : "오피스에 대한 설명"
        }}
        이때 주어지는 데이터는 위와 같이 name과 description이야

        {officeData}

        위 데이터들을 아래의 조건에 맞게 분류해줘
        이때 정확하게 아래 JSON 형태로 결과를 출력해줘
        1. 모니터 유무
        {{
            "filter" : "Monitor",
            "data" : [
                {{
                    "status" : "Y"
                    "names" : ["",""...]
                }},
                {{
                    "status" : "N"
                    "names" : ["",""...]
                }},
            ]
        }}
        2. 회의실 유무
        {{
            "filter" : "ConferenceRoom",
            "data" : [
                {{
                    "status" : "Y"
                    "names" : ["",""...]
                }},
                {{
                    "status" : "N"
                    "names" : ["",""...]
                }},
            ]
        }}
        3. 카페형 오피스 / 공유형 오피스 / 미분류
        {{
            "filter" : "OfficeType",
            "data" : [
                {{
                    "status" : "cafe"
                    "names" : ["",""...]
                }},
                {{
                    "status" : "shared"
                    "names" : ["",""...]
                }},
                {{
                    "status" : "N/A"
                    "names" : ["",""...]
                }},
            ]
        }}
        4. 주차 공간 유무
        {{
            "filter" : "Parking",
            "data" : [
                {{
                    "status" : "Y"
                    "names" : ["",""...]
                }},
                {{
                    "status" : "N"
                    "names" : ["",""...]
                }},
            ]
        }}
        5. 폰부스 유무
        {{
            "filter" : "PhoneBooth",
            "data" : [
                {{
                    "status" : "Y"
                    "names" : ["",""...]
                }},
                {{
                    "status" : "N"
                    "names" : ["",""...]
                }},
            ]
        }}

        이때 아래와 같은 조건을 지켜서 결과를 제공해줘
        a. JSON 형태는 내가 제공한 조건 그대로 바꾸지 말고 분류해줘
        b. 설명 없이 JSON 결과만 제공해줘
        c. 만약 3번 카페형 오피스 / 공유형 오피스 / 미분류 필터는 언급이 없다면 미분류로 해줘
        d. 만약 3번을 제외한 각 filter에 대한 언급이 없으면 "status" : "N"로 분류해줘
        e. 누락없이 모든 오피스를 분류해줘
    """)

    chain = (
        prompt 
        | llm 
        | StrOutputParser()
    )

    filter_result = await asyncio.to_thread(
        chain.invoke, {
               "officeData" : officeData,
        })
    
    print("\n[LangChain]-[filter_office] ", filter_result)

    json_result = extract_and_parse_json(filter_result)

    if not json_result:
        return None
    
    return json_result

######## Create Proposal ########
async def create_proposal(description):
    print("\n[LangChain]-[create_proposal] Description :", description)

    prompt = ChatPromptTemplate.from_template("""
        너는 현재 워케이션을 신청하고자 하는 직장인이야

        또한 워케이션에 대한 정보는 아래와 같아
        {description}

        이러한 워케이션 프로그램의 제안서의 내용은 아래와 같아

        1) 지원동기
        - 지원동기의 적절성 및 충실성이 포함되는 내용이 들어가야 해
        - 해당 지역의 특징 등 이해도 및 여행의지를 포함해서 작성해줘
        - 해당 지역에 정착하여 살 의향, 계획에 대해 자세하게 작성해줘
        - 해당 워케이션의 프로그램, 브랜드에 관한 내용을 포함해줘

        2) 여행계획
        - 여행일정의 구체성 / 여행기간 및 여행일정을 토대로 구체적으로 작성해줘야 해
        - 여행 기간은 1주일 이내의 기간으로 0일차의 형식으로 작성해줘
        - 해당 워케이션의 프로그램, 브랜드에 관한 내용을 포함해줘

        3) 홍보계획
        - SNS 팔로우 수 등 고려 홍보 효과성을 생각하여 홍보방법의 적절성(시기, 횟수, 매체 등 ) 에 대한 내용이 포함해줘

        이때 제안서의 내용에 맞게 아래에 주어진 모든 조건들을 반드시 맞추어서 모든 content value값이 한국어인 JSON을 만들어줘
        {{
            "name" : "1) 지원동기",
            "content" : "content value"
        }}
        {{
            "name" : "2) 여행계획",
            "content" : "content value"
        }}
        {{
            "name" : "3) 홍보계획",
            "content" : "content value"
        }}

        a. content value에 각 제안서 항목의 내용을 작성해서 넣어줘
        b. content value는 반드시 200자 이상 300자 내외의 한국어로 작성해줘
        c. 절대로 content value 항목의 내용 안에서 프로그램, 브랜드, 내용이나 구문, 표현을 반복해서 작성하지 말아줘
        d. JSON의 모든 content value값을 한국어로 작성해야해
    """)

    chain = (
        prompt 
        | llm 
        | StrOutputParser()
    )

    filter_result = await asyncio.to_thread(
        chain.invoke, {
               "description" : description,
        })

    print("\n[LangChain]-[create_proposal] ", filter_result)

    json_result = extract_and_parse_json(filter_result)

    if not json_result:
        return None
    
    return json_result