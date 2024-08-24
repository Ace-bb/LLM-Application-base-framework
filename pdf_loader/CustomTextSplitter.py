from typing import Callable
from conf import *
from langchain.text_splitter import TextSplitter
import copy
import logging
from abc import ABC, abstractmethod
from typing import (
    AbstractSet,
    Any,
    Callable,
    Collection,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)


class CustomTextSplitter(TextSplitter):
    def __init__(
            self, 
            encoding_name: str = "gpt2",
            model_name: Optional[str] = None,
            allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
            disallowed_special: Union[Literal["all"], Collection[str]] = "all",
            chunk_size: int = 4000, 
            chunk_overlap: int = 200, 
            length_function: Callable[[str], int] = ...,
            separators = ['。\n', '\n\n', '。', "\n", " ", ""]
        ):
        super().__init__(chunk_size, chunk_overlap, length_function)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 64,
            chunk_overlap  = 0,
            length_function = len,
            separators = ['。\n', '\n\n', '。', "\n", " ", ""]
        )
        try:
            import tiktoken
        except ImportError:
            raise ValueError(
                "Could not import tiktoken python package. "
                "This is needed in order to for TokenTextSplitter. "
                "Please install it with `pip install tiktoken`."
            )

        if model_name is not None:
            enc = tiktoken.encoding_for_model(model_name)
        else:
            enc = tiktoken.get_encoding(encoding_name)
        self._tokenizer = enc
        self._allowed_special = allowed_special
        self._disallowed_special = disallowed_special
    
    def encode_table(self, table):
        return self._tokenizer.encode(
            table,
            allowed_special=self._allowed_special,
            disallowed_special=self._disallowed_special,
        )
    
    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        splits = []
        chunks = self.text_splitter.split_text(text)
        merge_chunk = ''
        for chunk in chunks:
            token_len = self._tokenizer.encode(
                merge_chunk+chunk,
                allowed_special=self._allowed_special,
                disallowed_special=self._disallowed_special,
            )
            if token_len <= self._chunk_size:
                merge_chunk += chunk
            else:
                splits.append(merge_chunk)
                merge_chunk = f"{merge_chunk[-self._chunk_overlap:]} {chunk}"
        splits.append(self._tokenizer.encode(
            merge_chunk,
            allowed_special=self._allowed_special,
            disallowed_special=self._disallowed_special,
        ))
        return splits, chunks[-1]

        input_ids = self._tokenizer.encode(
            text,
            allowed_special=self._allowed_special,
            disallowed_special=self._disallowed_special,
        )
        start_idx = 0
        cur_idx = min(start_idx + self._chunk_size, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]
        while start_idx < len(input_ids):
            splits.append(self._tokenizer.decode(chunk_ids))
            start_idx += self._chunk_size - self._chunk_overlap
            cur_idx = min(start_idx + self._chunk_size, len(input_ids))
            chunk_ids = input_ids[start_idx:cur_idx]
        return splits