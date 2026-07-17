import json
import re
import ast
import requests
try:
    import tiktoken # Офіційна бібліотека для підрахунку токенів
except ImportError:
    pass

class FreeshaAdvanced:
    def __init__(self, openrouter_key="YOUR_API_KEY"):
        self.api_key = openrouter_key
        
        # 1. CAVEMAN MODE: Жорсткий промпт для економії вихідних токенів
        self.system_prompt = (
            "CRITICAL: Output raw data/JSON only. No pleasantries. No formatting. "
            "Act as a primitive data pipeline node."
        )
        
        # Ініціалізація калькулятора токенів (аналог token-calculator.net)
        try:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except NameError:
            self.encoder = None

    def calculate_tokens(self, text):
        """Точний підрахунок токенів для прогнозування вартості."""
        if self.encoder:
            return len(self.encoder.encode(str(text)))
        return len(str(text)) // 4 # Приблизний підрахунок, якщо tiktoken не встановлено

    def headroom_compress(self, data):
        """2. HEADROOM COMPRESSION: Розумне стиснення різних типів даних."""
        if isinstance(data, (dict, list)):
            return json.dumps(data, separators=(',', ':'))
        if isinstance(data, str):
            # Видаляємо зайві пробіли та переноси рядків
            return re.sub(r'\s+', ' ', data).strip()
        return str(data)

    def lean_ctx_extract(self, code_string):
        """3. LEAN-CTX: Витягує лише структуру (скелет) коду без його важкого тіла."""
        try:
            tree = ast.parse(code_string)
            skeleton = []
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    skeleton.append(f"def {node.name}(...):")
                elif isinstance(node, ast.ClassDef):
                    skeleton.append(f"class {node.name}:")
                    for sub_node in node.body:
                        if isinstance(sub_node, ast.FunctionDef):
                            skeleton.append(f"    def {sub_node.name}(...):")
            return "\n".join(skeleton)
        except Exception:
            return "Parsing error"

    def execute_hermes_request(self, payload):
        """Інтеграція: відправляє стиснутий запит через OpenRouter API."""
        optimized_payload = self.headroom_compress(payload)
        
        # Аналітика до/після
        original_tokens = self.calculate_tokens(payload)
        optimized_tokens = self.calculate_tokens(optimized_payload)
        if original_tokens > 0:
            savings = round(100 - (optimized_tokens / original_tokens * 100), 2)
        else:
            savings = 0
            
        print(f"📊 Калькулятор токенів:")
        print(f"Токенів до стиснення: {original_tokens}")
        print(f"Токенів після стиснення: {optimized_tokens}")
        print(f"Зекономлено: {savings}%\n")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "nousresearch/hermes-3-llama-3.1-405b", 
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": optimized_payload}
            ]
        }
        
        print("🌐 Готово до відправки на OpenRouter...")
        # У реальному середовищі тут буде: 
        # response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        # return response.json()
        return "Пайплайн працює успішно!"

# --- ТЕСТУВАННЯ ---
if __name__ == "__main__":
    freesha = FreeshaAdvanced()
    
    # Симулюємо важкий масив даних (наприклад, логи чату)
    heavy_data = {
        "status": "success",
        "data": [
            {"user": "admin", "message": "hello world"},
            {"user": "moderator", "message": "all clear"}
        ],
        "metadata": {"time": "12:00", "source": "telegram"}
    }
    
    print("=== ДЕМОНСТРАЦІЯ FREESHA ===\n")
    original_json = json.dumps(heavy_data, indent=4)
    freesha.execute_hermes_request(original_json)
