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
        1. 모니터 유무
        {{
            "filter" : "monitor"
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
            "filter" : "conference"
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
            "filter" : "type"
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
            "filter" : "park"
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
            "filter" : "phone"
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
        e. JSON을 출력할 때 각 filter의 제목은 생략하고 들여쓰기를 해줘
        f. 누락없이 모든 오피스를 분류해줘
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

    json_result = parse_JSON(filter_result, True)

    if not json_result:
        return None
    
    return json_result