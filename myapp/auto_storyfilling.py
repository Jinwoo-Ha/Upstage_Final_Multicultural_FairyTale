import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_upstage import ChatUpstage

@csrf_exempt
@require_http_methods(["POST"])
def auto_fill(request):
    try:
        print("Received request body:", request.body)
        
        data = json.loads(request.body)
        transcript = data.get('transcript', '')

        print("Extracted transcript:", transcript)

        if not transcript:
            return JsonResponse({"error": "No transcript provided"}, status=400)

        filled_form = auto_fill_form(transcript)
        
        print("Filled form result:", filled_form)

        return JsonResponse(filled_form)

    except json.JSONDecodeError:
        print("Failed to parse request body as JSON")
        return JsonResponse({"error": "Invalid JSON in request"}, status=400)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return JsonResponse({"error": str(e)}, status=500)

def auto_fill_form(user_info):
    form_fields = [
        {"Field Name": "이름", "Description": "주어진 문장에서 언급된 사람의 이름"},
        {"Field Name": "나이", "Description": "사람의 나이 (정수 형태로 반환, 예: 5)"},
        {"Field Name": "관심_있는_나라", "Description": "사람이 가고 싶어하는 나라"},
        {"Field Name": "좋아하는_것들", "Description": "사람이 좋아하는 것들 (쉼표로 구분된 목록)"},
    ]

    form_filling_prompt_template = ChatPromptTemplate.from_messages([
        (
            "human",
            """주어진 정보를 바탕으로 폼을 작성해주세요. 각 필드에 대한 값만 JSON 형식으로 반환해주세요.
            예를 들어, 다음과 같은 형식으로 반환해주세요:
            {{
                "이름": "홍길동",
                "나이": 25,
                "관심_있는_나라": "미국",
                "좋아하는_것들": "영화, 독서"
            }}
            나이는 반드시 정수로 반환해주세요.
            알 수 없는 정보는 빈 문자열로 남겨두세요.

            ---
            **개인 정보:**
            {my_info}
            ---
            **작성할 폼:**
            {form_fields}
            """
        )
    ])

    solar_mini = ChatUpstage(model="solar-1-mini-chat")
    chain_mini = form_filling_prompt_template | solar_mini | StrOutputParser()

    filled_form = chain_mini.invoke(
        {
            "my_info": user_info,
            "form_fields": form_fields,
        }
    )

    # 결과를 파싱하여 딕셔너리로 변환
    try:
        form_data = json.loads(filled_form)
        # 나이를 정수로 변환
        if '나이' in form_data and form_data['나이']:
            form_data['나이'] = int(form_data['나이'])
        return form_data
    except json.JSONDecodeError:
        print("Failed to parse AI response as JSON. Raw content:", filled_form)
        return {"error": "Invalid form data generated"}