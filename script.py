import json
import re
from typing import Dict, List, Any
from datetime import datetime
from groq import Groq
import tiktoken
import time
from dotenv import load_dotenv
import os

load_dotenv()


class LLMEvaluationPipeline:
    def __init__(
        self,
        llm_api_key: str,
        llm_model: str,
        input_token_price: float,
        output_token_price: float,
        system_prompt: str = None,
    ):
        self.client = Groq(api_key=llm_api_key)
        self.llm_model = llm_model
        self.input_token_price = input_token_price
        self.output_token_price = output_token_price
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        if system_prompt:
            self._system_prompt = system_prompt
        else:
            self._system_prompt = """
                You are an AI Reliability Auditor. Evaluate the AI Response based strictly on the Context provided.
                
                Output JSON format:
                {
                    "relevance_score": (0.0 to 1.0), 
                    "factual_accuracy_score": (0.0 to 1.0),
                    "hallucination_detected": boolean,
                    "reasoning": "Concise explanation of the score."
                }

                Rules for 'factual_accuracy_score':
                - If the AI mentions specific numbers, prices, or policies (e.g., "Rs 2000") that appear nowhere in the Context, score is 0.0.
                - If the AI says "I don't know" when the answer is missing, score is 1.0.
            """

    def _calculate_latency(self, user_created_at: str, ai_updated_at: str) -> Any:
        try:
            format_str = "%Y-%m-%dT%H:%M:%S.%fZ"
            t1 = datetime.strptime(user_created_at, format_str)
            t2 = datetime.strptime(ai_updated_at, format_str)
            latency = (t2 - t1).total_seconds()
            return latency
        except Exception as e:
            print(f"error in latency calculation: {str(e)}")
            return e

    def _gemini_as_judge(self, content: str) -> Dict:
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": content},
                ],
            )
            res = response.choices[0].message.content
            cleaned_response = re.sub(r"```.*?\n", "", res).strip()
            cleaned_response = cleaned_response.replace("```", "").strip()

            return json.loads(cleaned_response)
        except Exception as e:
            raise e

    def _estimate_cost(self, input_tokens, output_tokens) -> Dict:
        try:
            input_cost = input_tokens * self.input_token_price
            output_cost = output_tokens * self.output_token_price
            total_cost = input_cost + output_cost

            return total_cost
        except Exception as e:
            print(f"error in cost estimation: {str(e)}")
            return "Unable to estimate cost"

    def _evaluate(
        self,
        user_turn: Dict,
        ai_turn: Dict,
        ground_truth_context: List,
        context_token: int,
    ) -> Dict:
        try:
            # 1. Calculating Latency
            latency = self._calculate_latency(
                user_turn.get("created_at"), ai_turn.get("created_at")
            )

            # 2. Using LLM-as-judge
            user_content = f"""
                ___CONTEXT (GROUND TRUTH)___:
                {ground_truth_context}

                ___USER QUERY___:
                {user_turn['message']}

                ___AI RESPONSE___:
                {ai_turn['message']}
            """
            llm_eval_response = self._gemini_as_judge(user_content)

            # 3. Estimating input and output token cost
            input_tokens = (
                len(self.tokenizer.encode(user_turn["message"])) + context_token
            )
            output_tokens = len(self.tokenizer.encode(ai_turn["message"]))
            cost_estimation = self._estimate_cost(
                input_tokens=input_tokens, output_tokens=output_tokens
            )

            input_tokens_for_eval = len(self.tokenizer.encode(user_content))
            output_tokens_for_eval = len(
                self.tokenizer.encode(llm_eval_response.__str__())
            )
            evaluation_cost = self._estimate_cost(
                input_tokens=input_tokens_for_eval, output_tokens=output_tokens_for_eval
            )

            evaluation_result = {
                "evaluation_timestamp": datetime.now().isoformat(),
                "target_turn": ai_turn.get("turn"),
                "metrics": {
                    "latency_seconds": latency,
                    "conversation_cost_usd": cost_estimation,
                    "eval_cost_usd": evaluation_cost,
                },
                "quality_check": llm_eval_response,
            }
            return evaluation_result
        except Exception as e:
            print(f"error in evaluation: {str(e)}")
            return {}

    def evaluate_ai_reponse(
        self,
        chat_conversation: Dict,
        context_data: Dict,
        ai_turn_id=None,
        evaluate_all: bool = False,
    ) -> None:
        try:
            if ai_turn_id is None and not evaluate_all:
                raise Exception("Either ai_turn_id or evaluate_all must be provided.")

            turns = chat_conversation.get("conversation_turns")
            if turns is None:
                raise Exception("No turns found in the conversation.")

            context = context_data.get("data").get("sources").get("vectors_used")
            if context is None:
                raise Exception("No used context vector id found in data.")

            vectors_data = context_data.get("data").get("vector_data")
            if vectors_data is None:
                raise Exception("No vector data found.")

            used_context = list()
            used_context_tokens = 0

            for v in vectors_data:
                if v.get("id") in context:
                    used_context.append(
                        {"source_url": v.get("source_url"), "text": v.get("text")}
                    )
                    used_context_tokens += v.get("tokens")

            result = list()
            for index, turn in enumerate(turns):
                if (
                    ai_turn_id is not None
                    and turn.get("turn") == ai_turn_id
                    and turn.get("role") == "AI/Chatbot"
                ):
                    user_turn = turns[index - 1] if index > 0 else None
                    if user_turn is None:
                        raise Exception(
                            f"User turn before turn {ai_turn_id} not found."
                        )
                    else:
                        evaluation = self._evaluate(
                            user_turn, turn, used_context, used_context_tokens
                        )
                        # print(f"Evaluation for AI Turn {ai_turn_id}: {evaluation}")
                        result.append(evaluation)
                        break
                elif evaluate_all and turn.get("role") == "AI/Chatbot":
                    user_turn = turns[index - 1] if index > 0 else None
                    if user_turn is None:
                        print(
                            f"User turn before turn {turn.get('turn')} not found. Skipping evaluation for this turn."
                        )
                        continue
                    else:
                        evaluation = self._evaluate(
                            user_turn, turn, used_context, used_context_tokens
                        )
                        result.append(evaluation)
                        time.sleep(
                            1
                        )  # for avoiding free rate limits, you can remove it if you have paid plan
                else:
                    continue

            if not result:
                if evaluate_all is False:
                    print(f"No evaluation found for AI Turn ID: {ai_turn_id}")
                else:
                    print("No AI turns found for evaluation.")

            return result
        except Exception as e:
            print(f"error: {str(e)}")
            return


if __name__ == "__main__":
    with open("sample-chat-conversation-01.json", "r") as file:
        chat_conversation = json.load(file)
    with open("sample_context_vectors-01.json", "r") as file:
        context_data = json.load(file)

    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")
    input_price = float(os.getenv("INPUT_TOKEN_PRICE"))  # per 1M input tokens
    output_price = float(os.getenv("OUTPUT_TOKEN_PRICE"))  # per 1M output tokens

    pipeline = LLMEvaluationPipeline(
        llm_api_key=api_key,
        llm_model=model,
        input_token_price=input_price,
        output_token_price=output_price,
    )
    # result = pipeline.evaluate_ai_reponse(chat_conversation, context_data, ai_turn_id=6)
    result = pipeline.evaluate_ai_reponse(
        chat_conversation, context_data, evaluate_all=True
    )

    with open("evaluation.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        print("Evaluation results saved to evaluation.json")
