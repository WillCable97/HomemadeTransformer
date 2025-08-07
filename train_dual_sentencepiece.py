#!/usr/bin/env python3
"""
Script to train dual SentencePiece tokenizers for English-Spanish translation.
Extracts English and Spanish data from parallel corpus and trains separate models.
"""

import os
import sentencepiece as spm
from pathlib import Path

def extract_parallel_data(input_file: str):
    """
    Extract English and Spanish sentences from parallel corpus.
    
    Args:
        input_file: Path to the tab-separated parallel corpus file
        
    Returns:
        tuple: (english_sentences, spanish_sentences)
    """
    english_sentences = []
    spanish_sentences = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if '\t' in line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    eng = parts[0].strip()
                    spa = parts[1].strip().split('CC-BY')[0].strip()  # Remove attribution
                    
                    if eng and spa:  # Only add non-empty sentences
                        english_sentences.append(eng)
                        spanish_sentences.append(spa)
            else:
                print(f"Warning: Line {line_num} doesn't contain tab separator: {line[:50]}...")
    
    print(f"Extracted {len(english_sentences)} English-Spanish sentence pairs")
    return english_sentences, spanish_sentences

def write_sentences_to_file(sentences: list, output_file: str):
    """Write sentences to a file, one per line."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(sentence + '\n')
    print(f"Written {len(sentences)} sentences to {output_file}")

def train_sentencepiece_model(
    input_file: str,
    model_prefix: str,
    vocab_size: int = 8000,
    model_type: str = "bpe",
    input_sentence_size: int = 1000000,
    shuffle_input_sentence: bool = True,
    character_coverage: float = 0.9995,
    num_threads: int = 4
):
    """
    Train a SentencePiece model on the given text file.
    
    Args:
        input_file: Path to the input text file
        model_prefix: Prefix for the output model files
        vocab_size: Size of the vocabulary
        model_type: Type of model ('bpe', 'unigram', 'char', 'word')
        input_sentence_size: Number of sentences to use for training
        shuffle_input_sentence: Whether to shuffle input sentences
        character_coverage: Character coverage for the model
        num_threads: Number of threads to use
        
    Returns:
        str: Path to the trained model
    """
    
    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Create output directory if it doesn't exist
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    
    # Set up training parameters
    train_args = [
        f'--input={input_file}',
        f'--model_prefix={output_dir / model_prefix}',
        f'--vocab_size={vocab_size}',
        f'--model_type={model_type}',
        f'--input_sentence_size={input_sentence_size}',
        f'--shuffle_input_sentence={str(shuffle_input_sentence).lower()}',
        f'--character_coverage={character_coverage}',
        f'--num_threads={num_threads}',
        '--pad_id=0',
        '--unk_id=1',
        '--bos_id=2',
        '--eos_id=3',
        '--pad_piece=<PAD>',
        '--unk_piece=<UNK>',
        '--bos_piece=<SOS>',
        '--eos_piece=<EOS>'
    ]
    
    print(f"Training {model_prefix} SentencePiece model...")
    print(f"Input file: {input_file}")
    print(f"Model type: {model_type}")
    print(f"Vocabulary size: {vocab_size}")
    
    # Train the model
    spm.SentencePieceTrainer.train(' '.join(train_args))
    
    model_path = output_dir / f"{model_prefix}.model"
    vocab_path = output_dir / f"{model_prefix}.vocab"
    
    print(f"Training completed!")
    print(f"Model saved to: {model_path}")
    print(f"Vocabulary saved to: {vocab_path}")
    
    return str(model_path)

def test_sentencepiece_model(model_path: str, test_text: str, language: str):
    """Test the trained SentencePiece model."""
    try:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(model_path)
        
        print(f"\nTesting {language} model with text: '{test_text}'")
        tokens = sp.encode_as_pieces(test_text)
        token_ids = sp.encode_as_ids(test_text)
        
        print(f"Tokens: {tokens}")
        print(f"Token IDs: {token_ids}")
        print(f"Vocabulary size: {sp.get_piece_size()}")
        
        # Test decoding
        decoded = sp.decode_pieces(tokens)
        print(f"Decoded: '{decoded}'")
        
    except Exception as e:
        print(f"Error testing {language} model: {e}")

def main():
    """Main function to train dual SentencePiece models."""
    
    # Extract parallel data
    input_file = "data/raw/spa.txt"
    print(f"Extracting parallel data from {input_file}...")
    
    english_sentences, spanish_sentences = extract_parallel_data(input_file)
    
    # Write separate files for each language
    eng_file = "data/processed/english_sentences.txt"
    spa_file = "data/processed/spanish_sentences.txt"
    
    # Create processed directory
    Path("data/processed").mkdir(exist_ok=True)
    
    write_sentences_to_file(english_sentences, eng_file)
    write_sentences_to_file(spanish_sentences, spa_file)
    
    # Train English model
    print("\n" + "="*50)
    eng_model_path = train_sentencepiece_model(
        input_file=eng_file,
        model_prefix="english_sp",
        vocab_size=8000,
        model_type="bpe"
    )
    
    # Train Spanish model
    print("\n" + "="*50)
    spa_model_path = train_sentencepiece_model(
        input_file=spa_file,
        model_prefix="spanish_sp",
        vocab_size=8000,
        model_type="bpe"
    )
    
    # Test both models
    print("\n" + "="*50)
    test_sentencepiece_model(eng_model_path, "Hello world, how are you?", "English")
    test_sentencepiece_model(spa_model_path, "Hola mundo, ¿cómo estás?", "Spanish")
    
    print(f"\n" + "="*50)
    print("Training completed! You can now use:")
    print(f"English model: {eng_model_path}")
    print(f"Spanish model: {spa_model_path}")
    print("\nExample usage:")
    print("from homemadetransformer.tokenizers import SentencePieceTokenizer")
    print("src_tok = SentencePieceTokenizer('models/english_sp.model')")
    print("tgt_tok = SentencePieceTokenizer('models/spanish_sp.model')")

if __name__ == "__main__":
    main() 