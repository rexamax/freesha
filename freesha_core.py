import json
import re
import ast

class FreeshaOptimizer:
    def __init__(self):
        # 1. CAVEMAN MODE 
        # Forces the model to respond without politeness or unnecessary tokens
        self.system_prompt = (
            "CRITICAL: Output raw, strict data only. No pleasantries, no markdown formatting, "
            "no explanations. Act as a primitive, highly efficient data processor."
        )

    # 2. HEADROOM COMPRESSION
    def compress_payload(self, data):
        """Intelligently detects data type and compresses it to save input tokens."""
        if isinstance(data, dict) or isinstance(data, list):
            # Minify JSON aggressively by removing all unnecessary spaces and indents
            return json.dumps(data, separators=(',', ':'))
        elif isinstance(data, str):
            # Compress unstructured text (e.g., massive message logs)
            compressed = re.sub(r'\s+', ' ', data).strip()
            return compressed
        return str(data)

    # 3. LEAN-CTX (Contextual Skeleton)
    def extract_skeleton(self, code_string):
        """Parses Python code and returns only the structural skeleton (classes/functions) without the heavy body."""
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
        except Exception as e:
            return "Error parsing structural skeleton."

# --- DEMONSTRATION OF TOKEN SAVINGS ---
if __name__ == "__main__":
    freesha = FreeshaOptimizer()
    
    print("=== FREESHA MIDDLEWARE DEMO ===\n")
    
    # Scenario A: Compressing Massive Chat Archives
    chat_archive = {
        "source": "private_tg_channel_dump",
        "topic": "arbitrage_traffic_ubt",
        "total_messages": 1000000,
        "sample_batch": [
            {"user": "admin", "text": "Here is the new funnel for UBT...", "timestamp": "2019-05-12T10:00:00Z"},
            {"user": "media_buyer", "text": "Referral links are converting well. Need to set up domains.", "timestamp": "2019-05-12T10:05:00Z"}
        ]
    }
    
    original_json = json.dumps(chat_archive, indent=4)
    compressed_json = freesha.compress_payload(chat_archive)
    
    print("1. HEADROOM COMPRESSION (JSON Minification)")
    print(f"Original Payload Size: {len(original_json)} chars")
    print(f"Compressed Payload Size: {len(compressed_json)} chars")
    print(f"Token Savings: {round(100 - (len(compressed_json) / len(original_json) * 100), 2)}%\n")
    
    # Scenario B: Lean-ctx Code Skeleton Extraction for automated pipelines
    heavy_pipeline_code = '''
class RapidAPINewsParser:
    def __init__(self):
        self.interval_hours = 12
        
    def fetch_and_parse_data(self):
        # 100 lines of heavy logic, API requests, and data mapping...
        raw_data = "fetched_data"
        return raw_data
        
    def push_insights_to_telegram(self, insights):
        # 50 lines of Telegram Bot API integration...
        pass
'''
    print("2. LEAN-CTX (Structural Extraction for LLM context)")
    print("Sending only the necessary skeleton to the model:")
    print(freesha.extract_skeleton(heavy_pipeline_code))
