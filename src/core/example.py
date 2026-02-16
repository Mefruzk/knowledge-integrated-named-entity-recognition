from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from src.core.entity import Entity, EntitySource

@dataclass
class Example:
    """
    Represents a single biomedical document (typically PubMed abstract).
    
    Mutable container - entities can be added during processing pipeline:
    - Data creation: Add gold/dictionary entities
    - Model training: Add predicted entities
    - Inference: Add model predictions
    
    Unlike Entity (immutable), Examples evolve through the pipeline.
    
    Example usage:
        # Create from text
        example = Example(
            example_id="PMID_12345678",
            text="The EGFR protein is important...",
            title="EGFR in cancer"
        )
        
        # Add entities during processing
        example.add_entities([entity1, entity2], EntitySource.GOLD)
        example.add_entities([entity3], EntitySource.MODEL)
    """
    
    # Required fields
    example_id: str  # PMID or unique identifier
    text: str  # Abstract or document text
    
    # Optional fields
    title: Optional[str] = None
    
    # Entity collections (mutable - grow during pipeline)
    gold_entities: List[Entity] = field(default_factory=list)
    dictionary_entities: List[Entity] = field(default_factory=list)
    model_entities: List[Entity] = field(default_factory=list)
    agent_entities: List[Entity] = field(default_factory=list)
    
    # Extensible metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_entities(self, entities: List[Entity], source: EntitySource) -> None:
        """
        Add entities to appropriate list.
        
        Validates that all entities have matching source.
        
        Args:
            entities: List of Entity objects
            source: Which list to add to
            
        Raises:
            ValueError: If entity source doesn't match target source
        """
        # Validate source consistency
        for entity in entities:
            if entity.source != source:
                raise ValueError(
                    f"Entity source {entity.source} doesn't match target {source}"
                )
        
        # Add to appropriate list
        if source == EntitySource.GOLD:
            self.gold_entities.extend(entities)
        elif source == EntitySource.DICTIONARY:
            self.dictionary_entities.extend(entities)
        elif source == EntitySource.MODEL:
            self.model_entities.extend(entities)
        elif source == EntitySource.AGENT:
            self.agent_entities.extend(entities)
    
    def get_all_entities(
        self, 
        sources: Optional[List[EntitySource]] = None
    ) -> List[Entity]:
        """
        Get entities from specified sources.
        
        Args:
            sources: Which sources to include (None = all)
            
        Returns:
            Combined list of entities
        """
        if sources is None:
            sources = [
                EntitySource.GOLD, 
                EntitySource.DICTIONARY,
                EntitySource.MODEL, 
                EntitySource.AGENT
            ]
        
        entities = []
        for source in sources:
            if source == EntitySource.GOLD:
                entities.extend(self.gold_entities)
            elif source == EntitySource.DICTIONARY:
                entities.extend(self.dictionary_entities)
            elif source == EntitySource.MODEL:
                entities.extend(self.model_entities)
            elif source == EntitySource.AGENT:
                entities.extend(self.agent_entities)
        
        return entities
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary for storage.
        
        Returns:
            Dictionary representation
        """
        return {
            'example_id': self.example_id,
            'text': self.text,
            'title': self.title,
            'gold_entities': [e.model_dump() for e in self.gold_entities],
            'dictionary_entities': [e.model_dump() for e in self.dictionary_entities],
            'model_entities': [e.model_dump() for e in self.model_entities],
            'agent_entities': [e.model_dump() for e in self.agent_entities],
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Example':
        """
        Deserialize from dictionary.
        
        Args:
            data: Dictionary from to_dict()
            
        Returns:
            Example instance
        """
        return cls(
            example_id=data['example_id'],
            text=data['text'],
            title=data.get('title'),
            gold_entities=[Entity(**e) for e in data.get('gold_entities', [])],
            dictionary_entities=[Entity(**e) for e in data.get('dictionary_entities', [])],
            model_entities=[Entity(**e) for e in data.get('model_entities', [])],
            agent_entities=[Entity(**e) for e in data.get('agent_entities', [])],
            metadata=data.get('metadata', {})
        )