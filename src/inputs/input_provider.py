"""Abstract base class for all input providers - the core abstraction."""

from abc import ABC, abstractmethod
from typing import Iterator, Optional

from src.inputs.input_document import InputDocument


class InputProvider(ABC):
    """Abstract base class defining the contract for all input data sources.
    
    This class implements the Provider Pattern, a design pattern that defines
    a common interface for accessing data from various sources. All input
    providers (file readers, web scrapers, API clients, etc.) must implement
    this interface, ensuring consistent behavior across the codebase.
    
    How to Implement a New Provider
    ---------------------------------
    Subclass InputProvider and implement the required abstract methods:
    
    Example - File Provider:
        >>> from pathlib import Path
        >>> from typing import Iterator
        >>> 
        >>> class FileProvider(InputProvider):
        ...     def __init__(self, file_path: str):
        ...         self.file_path = Path(file_path)
        ...     
        ...     def __iter__(self) -> Iterator[InputDocument]:
        ...         with open(self.file_path) as f:
        ...             for line in f:
        ...                 # Parse line and create InputDocument
        ...                 doc = InputDocument(
        ...                     doc_id=f"doc_{line_num}",
        ...                     text=line.strip(),
        ...                     source="file"
        ...                 )
        ...                 yield doc
        ...     
        ...     def count(self) -> Optional[int]:
        ...         # File size known, return count
        ...         with open(self.file_path) as f:
        ...             return sum(1 for _ in f)
    
    Example - Streaming Web Provider:
        >>> class WebStreamProvider(InputProvider):
        ...     def __iter__(self) -> Iterator[InputDocument]:
        ...         # Stream from API endpoint
        ...         response = requests.get(api_url, stream=True)
        ...         for chunk in response.iter_lines():
        ...             doc = parse_chunk(chunk)
        ...             yield doc
        ...     
        ...     def count(self) -> Optional[int]:
        ...         # Unknown count for streaming
        ...         return None
    
    Real-World Usage
    ----------------
    The pipeline code doesn't need to know the source:
    
        >>> # Works with any provider
        >>> provider = FileProvider("data.txt")
        >>> # provider = PubTatorProvider(pmid_list)
        >>> # provider = WebFormProvider(form_data)
        >>> 
        >>> for doc in provider:
        ...     example = convert_to_example(doc)
        ...     process(example)
        >>> 
        >>> # Progress tracking
        >>> total = provider.count()
        >>> if total:
        ...     print(f"Processing {total} documents")
    
    This abstraction allows the same processing pipeline to work with:
    - Local files (JSON, TXT, BioC XML)
    - Remote APIs (PubTator, PubMed)
    - Web forms
    - Databases
    - Streams
    - Any future input source
    
    All without changing a single line of processing code.
    """
    
    @abstractmethod
    def __iter__(self) -> Iterator[InputDocument]:
        """Iterate over documents, yielding one InputDocument at a time.
        
        This method makes InputProvider instances iterable, allowing them
        to be used directly in for-loops and with iterator functions.
        
        Yields:
            InputDocument: The next document from the input source.
                Each document is yielded as soon as it's available,
                enabling memory-efficient streaming processing.
        
        Design Rationale:
        ----------------
        Using an iterator pattern rather than returning a list provides
        several key benefits:
        
        1. **Memory Efficiency**: Documents are processed one at a time,
           not all loaded into memory. Critical for large datasets.
        
        2. **Lazy Evaluation**: Documents are only fetched/parsed when
           needed. If processing stops early, unused documents aren't
           processed.
        
        3. **Streaming Support**: Natural fit for streaming sources
           (APIs, large files) where the total count may be unknown.
        
        4. **Pythonic**: Follows Python's iterator protocol, making
           the class work seamlessly with built-in functions like
           `next()`, `iter()`, and generator expressions.
        
        Example:
            >>> provider = SomeProvider(...)
            >>> for doc in provider:  # Uses __iter__
            ...     print(doc.doc_id)
            ...     process(doc)
        
        Note:
            Implementations should handle errors gracefully. If a document
            cannot be parsed, consider logging the error and continuing
            with the next document rather than raising an exception that
            stops the entire iteration.
        """
        pass
    
    @abstractmethod
    def count(self) -> Optional[int]:
        """Return the total number of documents available, if known.
        
        Returns:
            Optional[int]: The total count of documents if it can be
                determined, or None if the count is unknown (e.g., for
                streaming sources or when counting would be expensive).
        
        Design Rationale:
        ----------------
        This method serves multiple purposes:
        
        1. **Progress Tracking**: Allows progress bars and percentage
           calculations when processing documents.
        
        2. **Resource Planning**: Helps estimate processing time and
           resource requirements.
        
        3. **Validation**: Can verify expected document counts match
           actual counts.
        
        Why Optional?
        ------------
        Not all sources can provide a count efficiently:
        
        - **Known Count**: File providers can count lines, database
          queries can return COUNT(*), API responses may include
          total in headers.
        
        - **Unknown Count**: Streaming APIs, infinite generators,
          or sources where counting requires full iteration (which
          defeats the purpose of lazy evaluation).
        
        Example - Known Count:
            >>> provider = FileProvider("data.txt")
            >>> total = provider.count()  # 1000
            >>> for i, doc in enumerate(provider):
            ...     progress = (i + 1) / total * 100
            ...     print(f"Progress: {progress:.1f}%")
        
        Example - Unknown Count:
            >>> provider = StreamingAPIProvider(url)
            >>> total = provider.count()  # None
            >>> for doc in provider:
            ...     process(doc)  # Can't show percentage, but can show count
            ...     processed += 1
        
        Implementation Notes:
        -------------------
        - If counting is expensive (requires iteration), consider
          caching the result after first call.
        - For streaming sources, return None rather than iterating
          just to count (defeats lazy evaluation).
        - If the source supports it, use metadata/headers to get
          count without iteration.
        """
        pass


