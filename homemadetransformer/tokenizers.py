from typing import List
import re


class CharacterTokenizer:
    """Character-level tokenizer that splits text into individual characters."""
    
    def __init__(self, sos_token="<SOS>", eos_token="<EOS>", pad_token="<PAD>", unk_token="<UNK>"):
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into characters."""
        return list(text.lower().strip())
    
    def add_special_tokens(self, tokens: List[str]) -> List[str]:
        """Add SOS and EOS tokens to a sequence."""
        return [self.sos_token] + tokens + [self.eos_token]


class SubwordTokenizer:
    """Simple subword tokenizer using basic heuristics."""
    
    def __init__(self, sos_token="<SOS>", eos_token="<EOS>", pad_token="<PAD>", unk_token="<UNK>"):
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using subword patterns."""
        # Simple subword tokenization - split on common patterns
        text = text.lower().strip()
        
        # Split on word boundaries, keeping punctuation separate
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
        
        # Further split long words (simple heuristic)
        final_tokens = []
        for token in tokens:
            if len(token) > 8:  # Split long words
                # Split on common subword patterns
                subwords = re.findall(r'[a-z]+|[A-Z][a-z]*|[0-9]+', token)
                final_tokens.extend(subwords)
            else:
                final_tokens.append(token)
                
        return final_tokens
    
    def add_special_tokens(self, tokens: List[str]) -> List[str]:
        """Add SOS and EOS tokens to a sequence."""
        return [self.sos_token] + tokens + [self.eos_token]


class SentencePieceTokenizer:
    """Wrapper for SentencePiece tokenizer (requires sentencepiece library)."""
    
    def __init__(self, model_path: str, sos_token="<SOS>", eos_token="<EOS>", pad_token="<PAD>", unk_token="<UNK>"):
        try:
            import sentencepiece as spm
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(model_path)
            self.sos_token = sos_token
            self.eos_token = eos_token
            self.pad_token = pad_token
            self.unk_token = unk_token
        except ImportError:
            raise ImportError("sentencepiece library not found. Install with: pip install sentencepiece")
        
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using SentencePiece."""
        return self.sp.encode_as_pieces(text.lower().strip())
    
    def add_special_tokens(self, tokens: List[str]) -> List[str]:
        """Add SOS and EOS tokens to a sequence."""
        return [self.sos_token] + tokens + [self.eos_token]


class BPETokenizer:
    """Simple Byte Pair Encoding tokenizer implementation."""
    
    def __init__(self, vocab_size: int = 1000, sos_token="<SOS>", eos_token="<EOS>", pad_token="<PAD>", unk_token="<UNK>"):
        self.vocab_size = vocab_size
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.merges = {}
        self.vocab = {}
        
    def train(self, texts: List[str]):
        """Train the BPE tokenizer on a list of texts."""
        # This is a simplified BPE implementation
        # In practice, you'd want to use a more robust implementation
        
        # Initialize vocabulary with characters
        char_vocab = set()
        for text in texts:
            char_vocab.update(text.lower())
        
        self.vocab = {char: i for i, char in enumerate(char_vocab)}
        
        # Simple merge strategy (this is very basic)
        # In practice, you'd implement proper BPE algorithm
        for i in range(self.vocab_size - len(self.vocab)):
            # Find most frequent bigram and merge
            bigrams = {}
            for text in texts:
                text = text.lower()
                for j in range(len(text) - 1):
                    bigram = text[j:j+2]
                    bigrams[bigram] = bigrams.get(bigram, 0) + 1
            
            if not bigrams:
                break
                
            # Merge most frequent bigram
            most_freq_bigram = max(bigrams, key=bigrams.get)
            self.merges[most_freq_bigram] = len(self.vocab)
            self.vocab[most_freq_bigram] = len(self.vocab)
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using learned BPE merges."""
        text = text.lower().strip()
        tokens = list(text)
        
        # Apply merges
        for bigram, merge_id in self.merges.items():
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] + tokens[i+1] == bigram:
                    tokens[i:i+2] = [bigram]
                else:
                    i += 1
                    
        return tokens
    
    def add_special_tokens(self, tokens: List[str]) -> List[str]:
        """Add SOS and EOS tokens to a sequence."""
        return [self.sos_token] + tokens + [self.eos_token] 