import os
import google.generativeai as genai
import dotenv
import time
import openai

dotenv.load_dotenv()

class Gemini:
    def __init__(self, temperature=0):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel("gemini-2.0-flash")

    # @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def get_response(self, input):
        prompt = "\n\n".join([m["content"] for m in input])
        for i in range(10):
            try:
                response = self.model.generate_content(
                    prompt,
                    # generation_config=genai.types.GenerationConfig(
                    #     candidate_count=1,
                    #     max_output_tokens=2048,
                    #     temperature=0
                    # )
                )
                return response.text
            except Exception as e:
                print(f"Attempt {i+1} failed with error: {e}")
                time.sleep(2)

        return response.text[0]


class OpenAI:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("Must set environment variables for OPENAI_API_KEY")
        self.model = model
        self.temperature = temperature

    # @retry(
    #     wait=wait_random_exponential(min=1, max=60),
    #     stop=stop_after_attempt(5),
    #     reraise=True
    # )
    def get_response(self, messages: list[dict]) -> str:
        """
        messages: list các dict {role: "system"|"user"|"assistant", content: str}
        """
        resp = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=2048,
            n=1,
        )
        return resp.choices[0].message.content
