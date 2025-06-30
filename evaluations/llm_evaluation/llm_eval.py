"""
Acknowledgement:
    Based on the code from
    https://github.com/dominiksinsaarland/evaluating-topic-model-output
"""

import os
import json
import numpy as np
import random
import time
from llm_model import Gemini, OpenAI
import argparse
from tqdm import tqdm

os.makedirs("llm_eval_results", exist_ok=True)

parser = argparse.ArgumentParser(description="LLM evaluation")
parser.add_argument(
    "-r", "--tm_dataset_root_dir",
    type=str,
    default="tm_datasets",
    help="path to dataset",
)
parser.add_argument("-d", "--dataset_name", type=str, default="20NG")
parser.add_argument("-n", "--num_topics", type=int, default=20)
parser.add_argument("-t", "--num_topwords", type=int, default=15)
parser.add_argument(
    "-c", "--client", type=str, choices=["Gemini", "OpenAI"], default="OpenAI"
)
args = parser.parse_args()
beta_path = f"outputs/{args.dataset_name}_{args.num_topics}/beta.npy"

os.makedirs(f"llm_eval_results/{args.client}", exist_ok=True)

if args.client == "Gemini":
    llm_client = Gemini(temperature=1.0)
elif args.client == "OpenAI":
    llm_client = OpenAI(model="gpt-4o", temperature=1.0)
else:
    raise ValueError(f"Unknown client: {args.client}")


def get_system_prompt(dataset_name: str) -> str:
    """
    Return json system prompt for LLM evaluation of topic model output.
      {
        "rating": 1|2|3,
        "explanation": "Let’s think step by step: …"
      }
    """
    base = (
        "You are a helpful assistant evaluating the top words of a topic model output for a given topic. "
        "Please output **only** a JSON object with two keys:\n"
        '  - "rating": an integer 1, 2, or 3\n'
        '  - "explanation": a reasoning string that **starts** with "Let’s think step by step: "\n'
        "The JSON must be the only content in your response.\n"
    )
    if dataset_name == "20NG":
        return base + (
            "The topic modeling is based on the 20 Newsgroups dataset, a collection of approximately "
            "20,000 newsgroup posts across 20 forums."
        )
    elif dataset_name == "BBC_new":
        return base + (
            "The topic modeling is based on the BBC News dataset (2,225 articles in business, entertainment, politics, sport, tech)."
        )
    elif dataset_name == "NYT":
        return base + (
            "The topic modeling is based on the New York Times Annotated Corpus (articles from 1987–2007)."
        )
    elif dataset_name.startswith("WOS"):
        return base + (
            "The topic modeling is based on the Web of Science dataset (scholarly records across disciplines)."
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_system_prompt_non_reasoning(dataset_name: str) -> str:
    if dataset_name == "20NG":
        return """
You are a helpful assistant evaluating the top words of a topic model output for a given topic.
Please rate how related the following words are to each other on a scale from 1 to 3
("1" = not very related, "2" = moderately related, "3" = very related).
The topic modeling is based on the 20 Newsgroups dataset, comprising approximately 20,000 messages
from Usenet newsgroups organized into 20 categories such as comp.graphics, sci.space, rec.sport.baseball, etc.
Reply with a single number, indicating the overall appropriateness of the topic.
"""
    elif dataset_name == "BBC_new":
        return """
You are a helpful assistant evaluating the top words of a topic model output for a given topic.
Please rate how related the following words are to each other on a scale from 1 to 3
("1" = not very related, "2" = moderately related, "3" = very related).
The topic modeling is based on the BBC News dataset, containing 2,225 articles
classified into business, entertainment, politics, sport, and technology.
Reply with a single number, indicating the overall appropriateness of the topic.
"""
    elif dataset_name == "NYT":
        return """
You are a helpful assistant evaluating the top words of a topic model output for a given topic.
Please rate how related the following words are to each other on a scale from 1 to 3
("1" = not very related, "2" = moderately related, "3" = very related).
The topic modeling is based on The New York Times corpus. The corpus consists of articles from 1987 to 2007.
Sections from a typical paper include International, National, New York Regional, Business, Technology, and Sports news;
features on topics such as Dining, Movies, Travel, and Fashion; there are also obituaries and opinion pieces.
Reply with a single number, indicating the overall appropriateness of the topic.
"""
    elif dataset_name.startswith("WOS"):
        return """
You are a helpful assistant evaluating the top words of a topic model output for a given topic.
Please rate how related the following words are to each other on a scale from 1 to 3
("1" = not very related, "2" = moderately related, "3" = very related).
The topic modeling is based on the Web of Science dataset, a multidisciplinary corpus of scholarly records
including titles, abstracts, author keywords, and citations across fields such as biomedical sciences,
engineering, social sciences, and humanities.
Reply with a single number, indicating the overall appropriateness of the topic.
"""
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def llm_rating(
    tm_dataset_root_dir: str,
    beta_path: str, 
    dataset_name: str, 
    seed=42,
    num_words: int = 10,
    sample_k: int = 10,
    sleep_sec: float = 0.1,
):
    random.seed(seed)

    vocab_path = os.path.join(f"{tm_dataset_root_dir}/{dataset_name}", "vocab.txt")
    with open(vocab_path, "r", encoding="utf-8") as vf:
        vocab_list = [w.strip() for w in vf if w.strip()]
    vocab = {i: w for i, w in enumerate(vocab_list)}

    beta = np.load(beta_path)
    num_topics = beta.shape[0]
    print(num_topics)

    top_words = []
    for row in tqdm(beta):
        top_idxs = row.argsort()[::-1][:num_words]
        top_words.append([vocab[idx] for idx in top_idxs])

    system_prompt = get_system_prompt_non_reasoning(dataset_name)

    results = []
    for tid in tqdm(range(num_topics)):
        # if (tid + 1) % 15 == 0:
        #     time.sleep(60)
        words = top_words[tid].copy()
        random.shuffle(words)
        user_prompt = ", ".join(words)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = llm_client.get_response(messages)

        # if text is None:
        #     text = ""
        # text = text.strip()

        # # Handle code block markers (e.g., ```json ... ```)
        # if text.startswith("```json"):
        #     text = text[7:].strip()  # Remove ```json
        #     if text.endswith("```"):
        #         text = text[:-3].strip()  # Remove ```

        # Debug: Log the raw response for inspection
        print(f"Raw LLM response for topic {tid}: {text}")

        rating = 1  # Fallback rating
        explanation = "Let’s think step by step: Failed to parse LLM response or invalid response format."

        try:
            # Attempt to parse JSON
            # payload = json.loads(text)
            # if not isinstance(payload, dict):
            #     raise ValueError("Response is not a JSON object")

            # # Validate required keys
            # if "rating" not in payload or "explanation" not in payload:
            #     raise ValueError("Missing required keys: 'rating' or 'explanation'")

            # Validate rating
            rating_candidate = int(text)
            if not isinstance(rating_candidate, int) or rating_candidate not in [
                1,
                2,
                3,
            ]:
                raise ValueError(
                    f"Invalid rating: {rating_candidate}, expected 1, 2, or 3"
                )

            # # Validate explanation
            # explanation_candidate = payload["explanation"]
            # if not isinstance(explanation_candidate, str) or not explanation_candidate.startswith("Let’s think step by step: "):
            #     raise ValueError("Explanation must be a string starting with 'Let’s think step by step: '")

            rating = rating_candidate
            # explanation = explanation_candidate

        except (json.JSONDecodeError, ValueError) as e:
            # Log the error in the explanation but keep the fallback rating
            explanation = f"Let’s think step by step: Failed to parse LLM response. Error: {str(e)}. Response was: {text}"

        results.append(
            {
                "path": beta_path,
                "dataset_name": dataset_name,
                "topic_id": tid,
                "user_prompt": user_prompt,
                "rating": rating,  # Guaranteed to be 1, 2, or 3
                "explanation": explanation,
            }
        )
        time.sleep(sleep_sec)

    return results


if __name__ == "__main__":
    results = llm_rating(
        args.tm_dataset_root_dir, beta_path, args.dataset_name, args.num_topwords
    )

    with open(
        f"llm_eval_results/{args.client}/{args.dataset_name}_{args.num_topics}.json",
        "w",
    ) as f:
        json.dump(results, f, indent=4)
