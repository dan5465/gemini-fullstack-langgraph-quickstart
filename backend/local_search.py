import os

def search_in_directory(search_dir: str, queries: list[str]) -> list[str]:
    if not search_dir or not os.path.isdir(search_dir):
        return ["[No valid local directory provided]"]
    summaries = []
    for root, _, files in os.walk(search_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    if any(q.lower() in content.lower() for q in queries):
                        summaries.append(f"File: {file}\n{content[:500]}...")
            except Exception as e:
                summaries.append(f"Error reading {file}: {e}")
    return summaries
  #пошук винесено в окремий вузол, який можна замінити без змін у решті пайплайну та легко змінити джерело даних.Пошук можна замінити (ripgrep / embeddings / hybrid) без змін graph logic.Ізольована логіка пошуку з графіка та LLM за допомогою спеціального модуля пошуку
Покращений детермінізм, можливість використання в автономному режимі та тестованість дослідницького агента

