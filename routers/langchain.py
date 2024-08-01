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
    officeNum = len(officeData.offices)
    print(officeNum)

    prompt = ChatPromptTemplate.from_template("""
        너는 현재 기업들의 워케이션을 위한 데이터를 분류하는 담당자야
        {{
            "name" : "오피스의 이름"
            "description" : "오피스에 대한 설명"
        }}
        이때 주어지는 데이터는 위와 같이 name과 description이야

        {officeData}

        위 데이터들을 아래의 조건에 맞게 분류해줘
        이때 반드시 JSON 형태로 결과를 출력해줘
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
        e. 누락없이 모든 {officeNum}개의 오피스를 분류해줘
    """)

    chain = (
        prompt 
        | llm 
        | StrOutputParser()
    )

    filter_result = await asyncio.to_thread(
        chain.invoke, {
               "officeData" : officeData,
               "officeNum" : officeNum,
        })
    
    print("\n[LangChain]-[filter_office] ", filter_result)

    json_result = extract_and_parse_json(filter_result)

    if not json_result:
        return None
    
    return json_result


def proposal_parse_json(text):
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

######## Create Proposal ########
async def create_proposal(description, answer1, answer2, answer3, interest, job, purpose):
    print("\n[LangChain]-[create_proposal] Description\n", description)
    print("\n[LangChain]-[create_proposal] Answers \n", answer1, answer2, answer3)
    print("\n[LangChain]-[create_proposal] User Info \n", interest, job, purpose)

    prompt = ChatPromptTemplate.from_template("""
        너는 현재 워케이션을 신청하고자 하는 {job}이야
        너는 {interest}에 관심이 있고 {purpose}와 같은 목적을 가지고 워케이션을 가고자 해

        또한 워케이션에 대한 정보는 아래와 같아
        {description}

        이러한 워케이션 프로그램의 제안서의 내용은 아래와 같아

        1) 지원동기
        - 지원동기의 적절성 및 충실성이 포함되는 내용이 들어가야 해
        - 해당 지역의 특징 등 이해도 및 여행의지를 포함해서 작성해줘
        - 해당 지역에 정착하여 살 의향, 계획에 대해 자세하게 작성해줘
        - 해당 워케이션의 프로그램, 브랜드에 관한 내용을 포함해줘
        - 400자 이상 500자 내외

        2) 여행계획
        - 여행일정의 구체성 / 여행기간 및 여행일정을 토대로 구체적으로 작성해줘야 해
        - 여행 기간은 1주일 이내의 기간으로 0일차의 형식으로 작성해줘
        - 해당 워케이션의 프로그램, 브랜드에 관한 내용을 포함해줘
        - 400자 이상 500자 내외

        3) 홍보계획
        - SNS 팔로우 수 등 고려 홍보 효과성을 생각하여 홍보방법의 적절성(시기, 횟수, 매체 등 ) 에 대한 내용이 포함해줘
        - 400자 이상 500자 내외
                                              
        이때 제안서의 내용과 너의 직업, 관심사, 목적 맞게 아래에 주어진 모든 조건들을 반드시 맞추어서 모든 content value값이 한국어인 JSON을 만들어줘
        {{
            "name" : "1) 지원동기",
            "content" : content value
        }}
        1) 지원동기에 대해 미리 작성해둔 내용은 아래와 같아
        {answer1}
        {{
            "name" : "2) 여행계획",
            "content" : content value
        }}
        2) 여행계획에 대해 미리 작성해둔 내용은 아래와 같아
        {answer2}
        {{
            "name" : "3) 홍보계획",
            "content" : content value
        }}
        3) 홍보계획에 대해 미리 작성해둔 내용은 아래와 같아
        {answer3}
        
        a. 각 항목에 미리 작성해둔 내용을 바탕으로 content value값을 작성해줘
        b. content value는 모두 존댓말로 작성해줘
        c. content value에 각 제안서 항목의 내용을 작성해서 넣어줘
        d. content value는 반드시 400자 이상 500자 내외의 한국어로 작성해줘
        e. content value의 대명사는 '저는, 제가, 저의' 등으로 항상 고정해줘
        f. 절대로 content value 항목의 내용 안에서 프로그램, 브랜드, 내용이나 구문, 표현을 반복해서 작성하지 말아줘
        g. JSON의 모든 content value값을 한국어로 작성해야해
    """)

    chain = (
        prompt 
        | llm 
        | StrOutputParser()
    )

    filter_result = await asyncio.to_thread(
        chain.invoke, {
               "description" : description,
               "answer1" : answer1,
               "answer2" : answer2,
               "answer3" : answer3,
               "interest" : interest,
               "job" : job,
               "purpose" : purpose
        })
    
    print("\n[LangChain]-[create_proposal] ", filter_result)

    json_result = proposal_parse_json(filter_result)

    if not json_result:
        return None
    
    return json_result


######## Description LangChain ########
async def description_office(officeData):
    print("\n[LangChain]-[description_office] officeData :", officeData)
    officeNum = len(officeData.offices)
    print(officeNum)

    prompt = ChatPromptTemplate.from_template("""
        너는 현재 기업들의 워케이션을 위한 데이터를 정리하는 담당자야
        {{
            "name" : "오피스의 이름"
            "description" : "오피스에 대한 설명"
        }}
        이때 주어지는 데이터는 위와 같이 name과 description이야

        {officeData}

        각 데이터들을 아래의 조건에 맞게 정리해줘
        이때 반드시 JSON 배열 형태로 결과를 출력해줘
        {{
            "phoneNumber" : Value,
            "address" : Value,
            "operatingTime" : Value,
            "locationIntroduction" : Value,
            "providedDetails" : [
            ]
        }}
        phoneNumber의 Value에는 해당 장소의 전화번호를 적어줘
        address의 Value에는 해당 장소의 주소를 적어줘
        operatingTime의 Value에는 해당 장소의 운영시간을 적어줘
        locationIntroduction의 Value에는 해당 장소의 소개글을 적어줘
        providedDetails의 Value에는 위 데이터에서 정리되지 않는 데이터들을 정리해줘
        providedDetails의 Value에는 각 정리된 데이터를 별도의 분류 없이 String 값의 배열로만 집어넣어줘

        이때 아래와 같은 조건을 지켜서 결과를 제공해줘
        a. JSON 형태는 내가 제공한 조건 그대로 바꾸지 말고정리해줘
        b. 설명 없이 JSON 결과만 제공해줘
        c. 만약 각 Value에 대한 데이터가 없으면 Value에 null이라고 표시해줘
        d. 모든 Value 값은 한국어로 작성해줘
        e. 누락없이 모든 {officeNum}개 오피스의 name과 description를 분류해줘
    """)

    chain = (
        prompt 
        | llm 
        | StrOutputParser()
    )

    result = await asyncio.to_thread(
        chain.invoke, {
               "officeData" : officeData,
               "officeNum" : officeNum
        })
    
    print("\n[LangChain]-[description_office] ", result)

    json_result = proposal_parse_json(result)

    if not json_result:
        return None
    
    return json_result