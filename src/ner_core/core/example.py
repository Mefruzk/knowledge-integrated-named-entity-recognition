from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from ner_core.core.entity import Entity, EntitySource

@dataclass
class Example:
    """
    Represents a single biomedical document
    Mutable container - entities can be added during processing pipeline

    """
    # Required fields
    example_id: str  # PMID or unique identifier
    text: str  # Abstract or document text    
    # Optional fields
    title: Optional[str] = None

    entities_by_source: Dict[EntitySource, List[Entity]] = field(
        default_factory=lambda: {source: [] for source in EntitySource}
    )

    # Extensible metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_entities(self, entities: List[Entity], source: EntitySource) -> None:
        """Add entities to appropriate list."""

        for entity in entities:
            if entity.source != source:
                raise ValueError(f"Entity source {entity.source} doesn't match target {source}")

        self.entities_by_source[source].extend(entities)

    def get_all_entities(self, sources: Optional[List[EntitySource]] = None) -> List[Entity]:
        """Get all entities from specified sources."""
        if sources is None:
            sources = [s for s in EntitySource if s != EntitySource.OTHER]
        
        return [
            entity
            for source in sources
            for entity in self.entities_by_source[source]
        ]

    
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
            'entities_by_source': {
                source.value: [
                    {
                        **e.model_dump(),
                        'entity_id': str(e.entity_id)
                    }
                    for e in entities
                ]
                for source, entities in self.entities_by_source.items()
            },
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Example':
        """Deserialize from dictionary."""

        from uuid import UUID
        
        # Convert entity dicts back to Entity objects
        entities_by_source = {}
        for source_str, entities in data.get('entities_by_source', {}).items():
            source = EntitySource(source_str)
            entity_objects = []
            for e in entities:
                # Convert string back to UUID
                if isinstance(e.get('entity_id'), str):
                    e['entity_id'] = UUID(e['entity_id'])
                entity_objects.append(Entity(**e))
            entities_by_source[source] = entity_objects

        return cls(
            example_id=data['example_id'],
            text=data['text'],
            title=data.get('title'),
            entities_by_source=entities_by_source,
            metadata=data.get('metadata', {})

        )