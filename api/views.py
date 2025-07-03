from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView
from openai import OpenAI, RateLimitError
from os import getenv

model_mapping = {
    'deepseek-v3': "deepseek/deepseek-chat-v3-0324:free",
    'deepseek-r1': "deepseek/deepseek-r1:free",
    'gemini-2.0': "google/gemini-2.0-flash-exp:free",
    'qwen3': "qwen/qwen3-32b:free",
    'gemma-3': "google/gemma-3-27b-it:free",
}

def _get_model_name(model: str):
    if model not in model_mapping:
        raise ValueError()

    return model_mapping[model]


# Create your views here.
class AIView(APIView):
    http_method_names = ["post", "get"]

    def get(self, request):
        models = list(model_mapping.keys())
        return Response(models, status=status.HTTP_200_OK)

    def post(self, request):
        if not request.data:
            return Response(status=status.HTTP_400_BAD_REQUEST)

        message = request.data.get("message")
        model = request.data.get("model")

        if not message or not message.strip():
            return Response("Message field must be provided", status=status.HTTP_400_BAD_REQUEST)
        if not model or not model.strip():
            return Response("Model field must be provided", status=status.HTTP_400_BAD_REQUEST)

        try:
            model_q = _get_model_name(model)
        except ValueError:
            return Response(f"Model: {model} not found", status=status.HTTP_404_NOT_FOUND)

        try:
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=getenv("OPENROUTER_API_KEY"),
            )
        except Exception:
            return Response("Failed to initialize AI services", status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        try:
            completion = client.chat.completions.create(
                model=model_q,
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": message
                    },
                ],
            )

            if not completion.choices or not completion.choices[0].message.content:
                return Response("Received empty response from AI service", status=status.HTTP_502_BAD_GATEWAY)

            resp = {
                "response": completion.choices[0].message.content,
                "model_used": model,
            }

        except RateLimitError:
            return Response("Rate limit exceeded", status=status.HTTP_429_TOO_MANY_REQUESTS)
        except Exception:
            return Response("Unknown error occured", status=status.HTTP_503_SERVICE_UNAVAILABLE)

        return Response(resp, status=status.HTTP_200_OK)
