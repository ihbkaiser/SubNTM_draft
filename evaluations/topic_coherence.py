import os
import numpy as np
import tempfile


def TC_on_wikipedia(
    top_words: list[list[str]], cv_type: str = "C_V", jar_dir="evaluations", wiki_dir="evaluations"
):
    with tempfile.NamedTemporaryFile(
        mode="w+", delete=False, suffix=".txt", dir="/tmp"
    ) as tmp_file:
        top_word_path = tmp_file.name
        for topic in top_words:
            tmp_file.write(" ".join(topic) + "\n")

    random_number = np.random.randint(100000)

    tmp_output_path = f"/tmp/tmp{random_number}.txt"
    cmd = (
        f"java -jar {jar_dir}/palmetto.jar "
        f"{wiki_dir}/wikipedia_bd {cv_type} "
        f"{top_word_path} > {tmp_output_path}"
    )
    os.system(cmd)

    cv_score = []
    with open(tmp_output_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                score = float(parts[1])
                cv_score.append(score)
            except ValueError:
                print(f"[WARN] Skipped line (not a float): {line.strip()}")
                continue

    if len(cv_score) == 0:
        return [], 0.0

    return cv_score, sum(cv_score) / len(cv_score)
