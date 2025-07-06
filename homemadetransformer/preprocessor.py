import torch
from torch.utils.data import Dataset, DataLoader
from itertools import chain
from typing import List, Tuple, Dict, Optional
import re


class SimpleWordTokenizer:
    """Simple word-based tokenizer that splits on whitespace."""
    
    def __init__(self, sos_token="<SOS>", eos_token="<EOS>", pad_token="<PAD>", unk_token="<UNK>"):
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text by splitting on whitespace."""
        return text.lower().strip().split()
    
    def add_special_tokens(self, tokens: List[str]) -> List[str]:
        """Add SOS and EOS tokens to a sequence."""
        return [self.sos_token] + tokens + [self.eos_token]


class TranslationPreprocessor:
    """Preprocessor for translation datasets with configurable tokenization."""
    
    def __init__(self, 
                 src_tokenizer=None, 
                 tgt_tokenizer=None,
                 max_src_len: Optional[int] = None,
                 max_tgt_len: Optional[int] = None):
        """
        Initialize the preprocessor.
        
        Args:
            src_tokenizer: Tokenizer for source language (defaults to SimpleWordTokenizer)
            tgt_tokenizer: Tokenizer for target language (defaults to SimpleWordTokenizer)
            src_lang: Source language code
            tgt_lang: Target language code
            max_src_len: Maximum source sequence length (None for auto)
            max_tgt_len: Maximum target sequence length (None for auto)
        """
        self.src_tokenizer = src_tokenizer or SimpleWordTokenizer()
        self.tgt_tokenizer = tgt_tokenizer or SimpleWordTokenizer()
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        
        # Vocabulary mappings
        self.src_vocab = {}
        self.tgt_vocab = {}
        self.src_vocab_to_idx = {}
        self.tgt_vocab_to_idx = {}
        self.src_idx_to_vocab = {}
        self.tgt_idx_to_vocab = {}
        
        # Statistics
        self.src_vocab_size = 0
        self.tgt_vocab_size = 0
        self.max_src_len_actual = 0
        self.max_tgt_len_actual = 0
        
    def clean_translation_pair(self, line: str) -> Optional[Tuple[str, str]]:
        """Clean a translation pair from the dataset."""
        parts = line.strip().split("\t")
        
        if len(parts) < 2:
            return None
            
        src_text = parts[0].strip()
        tgt_text = parts[1].strip().split("CC-BY")[0].strip()  # Remove attribution text
        
        return src_text, tgt_text
    
    def build_vocabularies(self, src_sentences: List[str], tgt_sentences: List[str]):
        """Build vocabulary mappings from tokenized sentences."""
        # Tokenize all sentences
        src_tokenized = [self.src_tokenizer.add_special_tokens(self.src_tokenizer.tokenize(sent)) 
                        for sent in src_sentences]
        tgt_tokenized = [self.tgt_tokenizer.add_special_tokens(self.tgt_tokenizer.tokenize(sent)) 
                        for sent in tgt_sentences]
        
        # Build source vocabulary
        src_vocab = set(chain.from_iterable(src_tokenized))
        self.src_vocab_to_idx = {word: i for i, word in enumerate(src_vocab)}
        self.src_idx_to_vocab = {i: word for word, i in self.src_vocab_to_idx.items()}
        self.src_vocab_size = len(src_vocab)
        
        # Build target vocabulary
        tgt_vocab = set(chain.from_iterable(tgt_tokenized))
        self.tgt_vocab_to_idx = {word: i for i, word in enumerate(tgt_vocab)}
        self.tgt_idx_to_vocab = {i: word for word, i in self.tgt_vocab_to_idx.items()}
        self.tgt_vocab_size = len(tgt_vocab)
        
        # Calculate max lengths
        self.max_src_len_actual = max(len(s) for s in src_tokenized)
        self.max_tgt_len_actual = max(len(s) for s in tgt_tokenized)
        
        # Use provided max lengths if specified
        if self.max_src_len is not None:
            self.max_src_len_actual = min(self.max_src_len_actual, self.max_src_len)
        if self.max_tgt_len is not None:
            self.max_tgt_len_actual = min(self.max_tgt_len_actual, self.max_tgt_len)
            
        return src_tokenized, tgt_tokenized
    
    def encode_sentence(self, sentence: List[str], vocab_to_idx: Dict[str, int]) -> List[int]:
        """Encode a tokenized sentence to indices."""
        return [vocab_to_idx.get(word, vocab_to_idx.get(self.src_tokenizer.unk_token, 0)) 
                for word in sentence]
    
    def pad_sequence(self, seq: List[int], max_len: int, pad_idx: int = 0) -> List[int]:
        """Pad a sequence to the specified length."""
        return seq + [pad_idx] * (max_len - len(seq))
    
    def preprocess_data(self, raw_data: List[str]) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Preprocess raw translation data.
        
        Args:
            raw_data: List of tab-separated translation pairs
            
        Returns:
            Tuple of (encoded_src_sentences, encoded_tgt_sentences)
        """
        # Clean the data
        clean_pairs = [self.clean_translation_pair(line) for line in raw_data]
        clean_pairs = [pair for pair in clean_pairs if pair is not None]
        
        # Split into source and target
        src_sentences = [pair[0] for pair in clean_pairs]
        tgt_sentences = [pair[1] for pair in clean_pairs]
        
        # Build vocabularies and tokenize
        src_tokenized, tgt_tokenized = self.build_vocabularies(src_sentences, tgt_sentences)
        
        # Encode sentences
        src_encoded = [self.encode_sentence(sent, self.src_vocab_to_idx) for sent in src_tokenized]
        tgt_encoded = [self.encode_sentence(sent, self.tgt_vocab_to_idx) for sent in tgt_tokenized]
        
        # Pad sequences
        src_padded = [self.pad_sequence(sent, self.max_src_len_actual) for sent in src_encoded]
        tgt_padded = [self.pad_sequence(sent, self.max_tgt_len_actual) for sent in tgt_encoded]
        
        return src_padded, tgt_padded
    
    def get_vocab_info(self) -> Dict:
        """Get vocabulary information."""
        return {
            'src_vocab_size': self.src_vocab_size,
            'tgt_vocab_size': self.tgt_vocab_size,
            'max_src_len': self.max_src_len_actual,
            'max_tgt_len': self.max_tgt_len_actual,
            'src_vocab_to_idx': self.src_vocab_to_idx,
            'tgt_vocab_to_idx': self.tgt_vocab_to_idx,
            'src_idx_to_vocab': self.src_idx_to_vocab,
            'tgt_idx_to_vocab': self.tgt_idx_to_vocab
        }


class TranslationDataset(Dataset):
    """PyTorch Dataset for translation data."""
    
    def __init__(self, src_data: List[List[int]], tgt_data: List[List[int]]):
        self.src_data = src_data
        self.tgt_data = tgt_data
        
    def __len__(self):
        return len(self.src_data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.src_data[idx]), torch.tensor(self.tgt_data[idx])


def create_translation_dataloader(src_data: List[List[int]], 
                                tgt_data: List[List[int]], 
                                batch_size: int = 32, 
                                shuffle: bool = True) -> DataLoader:
    """Create a DataLoader for translation data."""
    dataset = TranslationDataset(src_data, tgt_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) 