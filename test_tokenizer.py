import sys
import io

# Force UTF-8 output on Windows consoles
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

encoded = tokenizer.encode("Hello, I am JED AI.")

print("Tokens:", encoded.tokens)
print("IDs:   ", encoded.ids)

# Round-trip decode test
decoded = tokenizer.decode(encoded.ids)
print("Decoded:", decoded)
print("Vocab size:", tokenizer.get_vocab_size())
print("\nAll OK!")