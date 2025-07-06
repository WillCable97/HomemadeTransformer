"""
Example usage of the TranslationPreprocessor with different tokenizers.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from homemadetransformer.preprocessor import TranslationPreprocessor, create_translation_dataloader
from homemadetransformer.tokenizers import CharacterTokenizer, SubwordTokenizer, SimpleWordTokenizer


def load_data(file_path: str):
    """Load translation data from file."""
    with open(file_path, encoding="utf-8") as f:
        lines = f.read().strip().split("\n")
    return lines


def main():
    # Load the data
    data = load_data("spa.txt")
    print(f"Loaded {len(data)} translation pairs")
    
    # Example 1: Using simple word tokenizer (original approach)
    print("\n=== Example 1: Simple Word Tokenizer ===")
    preprocessor = TranslationPreprocessor(
        src_tokenizer=SimpleWordTokenizer(),
        tgt_tokenizer=SimpleWordTokenizer()
    )
    
    src_data, tgt_data = preprocessor.preprocess_data(data)
    vocab_info = preprocessor.get_vocab_info()
    
    print(f"Source vocab size: {vocab_info['src_vocab_size']}")
    print(f"Target vocab size: {vocab_info['tgt_vocab_size']}")
    print(f"Max source length: {vocab_info['max_src_len']}")
    print(f"Max target length: {vocab_info['max_tgt_len']}")
    
    # Create dataloader
    dataloader = create_translation_dataloader(src_data, tgt_data, batch_size=32)
    print(f"Number of batches: {len(dataloader)}")
    
    # Example 2: Using character-level tokenizer
    print("\n=== Example 2: Character Tokenizer ===")
    char_preprocessor = TranslationPreprocessor(
        src_tokenizer=CharacterTokenizer(),
        tgt_tokenizer=CharacterTokenizer()
    )
    
    char_src_data, char_tgt_data = char_preprocessor.preprocess_data(data)
    char_vocab_info = char_preprocessor.get_vocab_info()
    
    print(f"Source vocab size: {char_vocab_info['src_vocab_size']}")
    print(f"Target vocab size: {char_vocab_info['tgt_vocab_size']}")
    print(f"Max source length: {char_vocab_info['max_src_len']}")
    print(f"Max target length: {char_vocab_info['max_tgt_len']}")
    
    # Example 3: Using subword tokenizer
    print("\n=== Example 3: Subword Tokenizer ===")
    subword_preprocessor = TranslationPreprocessor(
        src_tokenizer=SubwordTokenizer(),
        tgt_tokenizer=SubwordTokenizer()
    )
    
    subword_src_data, subword_tgt_data = subword_preprocessor.preprocess_data(data)
    subword_vocab_info = subword_preprocessor.get_vocab_info()
    
    print(f"Source vocab size: {subword_vocab_info['src_vocab_size']}")
    print(f"Target vocab size: {subword_vocab_info['tgt_vocab_size']}")
    print(f"Max source length: {subword_vocab_info['max_src_len']}")
    print(f"Max target length: {subword_vocab_info['max_tgt_len']}")
    
    # Example 4: Using different tokenizers for source and target
    print("\n=== Example 4: Mixed Tokenizers ===")
    mixed_preprocessor = TranslationPreprocessor(
        src_tokenizer=SimpleWordTokenizer(),
        tgt_tokenizer=CharacterTokenizer()
    )
    
    mixed_src_data, mixed_tgt_data = mixed_preprocessor.preprocess_data(data)
    mixed_vocab_info = mixed_preprocessor.get_vocab_info()
    
    print(f"Source vocab size: {mixed_vocab_info['src_vocab_size']}")
    print(f"Target vocab size: {mixed_vocab_info['tgt_vocab_size']}")
    print(f"Max source length: {mixed_vocab_info['max_src_len']}")
    print(f"Max target length: {mixed_vocab_info['max_tgt_len']}")
    
    # Example 5: Using length limits
    print("\n=== Example 5: With Length Limits ===")
    limited_preprocessor = TranslationPreprocessor(
        src_tokenizer=SimpleWordTokenizer(),
        tgt_tokenizer=SimpleWordTokenizer(),
        max_src_len=20,
        max_tgt_len=25
    )
    
    limited_src_data, limited_tgt_data = limited_preprocessor.preprocess_data(data)
    limited_vocab_info = limited_preprocessor.get_vocab_info()
    
    print(f"Source vocab size: {limited_vocab_info['src_vocab_size']}")
    print(f"Target vocab size: {limited_vocab_info['tgt_vocab_size']}")
    print(f"Max source length: {limited_vocab_info['max_src_len']}")
    print(f"Max target length: {limited_vocab_info['max_tgt_len']}")
    
    # Show some sample tokenizations
    print("\n=== Sample Tokenizations ===")
    sample_text = "Hello, how are you?"
    
    # Word tokenizer
    word_tokens = SimpleWordTokenizer().tokenize(sample_text)
    print(f"Word tokens: {word_tokens}")
    
    # Character tokenizer
    char_tokens = CharacterTokenizer().tokenize(sample_text)
    print(f"Character tokens: {char_tokens}")
    
    # Subword tokenizer
    subword_tokens = SubwordTokenizer().tokenize(sample_text)
    print(f"Subword tokens: {subword_tokens}")


if __name__ == "__main__":
    main() 