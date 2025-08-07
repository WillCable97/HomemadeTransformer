#!/usr/bin/env python3
"""
Script to train a SentencePiece tokenizer on Spanish text data.
"""

import os
import sentencepiece as spm
from pathlib import Path

def train_sentencepiece_model(
    input_file: str,
    model_prefix: str = "sp_model",
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
    
    print(f"Training SentencePiece model...")
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    print(f"Model type: {model_type}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Training arguments: {' '.join(train_args)}")
    
    # Train the model
    spm.SentencePieceTrainer.train(' '.join(train_args))
    
    model_path = output_dir / f"{model_prefix}.model"
    vocab_path = output_dir / f"{model_prefix}.vocab"
    
    print(f"\nTraining completed!")
    print(f"Model saved to: {model_path}")
    print(f"Vocabulary saved to: {vocab_path}")
    
    return str(model_path)

def test_sentencepiece_model(model_path: str, test_text: str = "Hola mundo"):
    """Test the trained SentencePiece model."""
    try:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(model_path)
        
        print(f"\nTesting model with text: '{test_text}'")
        tokens = sp.encode_as_pieces(test_text)
        token_ids = sp.encode_as_ids(test_text)
        
        print(f"Tokens: {tokens}")
        print(f"Token IDs: {token_ids}")
        print(f"Vocabulary size: {sp.get_piece_size()}")
        
        # Test decoding
        decoded = sp.decode_pieces(tokens)
        print(f"Decoded: '{decoded}'")
        
    except Exception as e:
        print(f"Error testing model: {e}")

if __name__ == "__main__":
    # Train the model
    input_file = "data/raw/spa.txt"
    model_path = train_sentencepiece_model(
        input_file=input_file,
        model_prefix="spanish_sp",
        vocab_size=8000,
        model_type="bpe"
    )
    
    # Test the model
    test_sentencepiece_model(model_path, "Hola mundo, ¿cómo estás?") 