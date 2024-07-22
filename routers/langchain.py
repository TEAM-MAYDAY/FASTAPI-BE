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


def extract_and_parse_json(text):
    # JSON 객체를 찾는 정규 표현식 패턴
    json_pattern = re.compile(r'\{[\s\S]*?\}(?=\s*\{|\s*$)')
    
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
            
            # 콤마 추가 및 대괄호로 감싸기
            json_str = json_str.replace('}\n{', '},{')
            if not json_str.strip().startswith('['):
                json_str = '[' + json_str + ']'
            
            # JSON 파싱
            parsed_json = json.loads(json_str)
            
            # 단일 객체인 경우 리스트에서 추출
            if isinstance(parsed_json, list) and len(parsed_json) == 1:
                parsed_json = parsed_json[0]
            
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
            "filter" : "Monitor"
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
            "filter" : "ConferenceRoom"
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
            "filter" : "OfficeType"
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
            "filter" : "Parking"
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
            "filter" : "PhoneBooth"
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

    json_result = extract_and_parse_json(filter_result)

    if not json_result:
        return None
    
    return json_result