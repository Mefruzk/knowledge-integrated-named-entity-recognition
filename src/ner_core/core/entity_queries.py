"""Utility class for querying entities from Example objects."""

from typing import List, Optional, Any
from uuid import UUID

from src.core.entity import Entity, EntitySource
from src.core.example import Example

class EntityQuery:
    """Utility class with static methods for querying entities from Example objects.
    
    All methods use Example.get_all_entities() internally and return filtered
    lists of Entity objects based on various criteria.
    """

    @staticmethod
    def by_entity_id(
        example: Example, 
        entity_id: str
    ) -> List[Entity]:
        """Find entities by their unique entity ID.
        
        Args:
            example: The Example object to search in
            entity_id: UUID string of the entity to find
            
        Returns:
            List containing the entity if found, empty list otherwise
            
        Example:
            >>> example = Example(example_id="PMID_123", text="EGFR is important")
            >>> entity = Entity(
            ...     start_offset=0, end_offset=4, span_text="EGFR",
            ...     label="PROTEIN", source=EntitySource.GOLD
            ... )
            >>> example.add_entities([entity], EntitySource.GOLD)
            >>> found = EntityQuery.by_entity_id(example, str(entity.entity_id))
            >>> len(found) == 1
            True
        """
        entities = example.get_all_entities()
        try:
            # Convert string to UUID for comparison
            target_uuid = UUID(entity_id)
            return [e for e in entities if e.entity_id == target_uuid]
        except ValueError:
            # Invalid UUID format
            return []

    @staticmethod
    def by_label(
        example: Example,
        label: str,
        sources: Optional[List[EntitySource]] = None
    ) -> List[Entity]:
        """Filter entities by their label/type.
        
        Args:
            example: The Example object to search in
            label: The entity label to filter by (e.g., "DISEASE", "PROTEIN")
            sources: Optional list of EntitySource values to filter by.
                    If None, searches all sources.
                    
        Returns:
            List of entities matching the label and source filter
        """
        entities = example.get_all_entities(sources)
        return [e for e in entities if e.label == label]

    @staticmethod
    def in_range(
        example: Example,
        start: int,
        end: int,
        sources: Optional[List[EntitySource]] = None
    ) -> List[Entity]:
        """Find entities that overlap with the specified character range.
        
        An entity overlaps if it has any character in common with the range [start, end).
        This includes entities that are fully contained, partially overlap, or
        fully contain the specified range.
        
        Args:
            example: The Example object to search in
            start: Starting character offset (inclusive)
            end: Ending character offset (exclusive)
            sources: Optional list of EntitySource values to filter by.
                    If None, searches all sources.
                    
        Returns:
            List of entities that overlap with the specified range
            
        """
        entities = example.get_all_entities(sources)
        overlapping = []
        for entity in entities:
            # Check if ranges overlap: entity overlaps if it's not completely
            # before or completely after the target range
            if not (entity.end_offset <= start or entity.start_offset >= end):
                overlapping.append(entity)
        return overlapping

    @staticmethod
    def by_confidence(
        example: Example,
        min_conf: float,
        sources: Optional[List[EntitySource]] = None
    ) -> List[Entity]:
        """Filter entities by minimum confidence threshold.
        
        Only entities with confidence >= min_conf are returned. Entities without
        a confidence score (None) are excluded.
        
        Args:
            example: The Example object to search in
            min_conf: Minimum confidence score (0.0 to 1.0)
            sources: Optional list of EntitySource values to filter by.
                    If None, searches all sources.
                    
        Returns:
            List of entities with confidence >= min_conf
        """
        if not (0.0 <= min_conf <= 1.0):
            raise ValueError("min_conf must be between 0.0 and 1.0")
        
        entities = example.get_all_entities(sources)
        return [
            e for e in entities 
            if e.confidence is not None and e.confidence >= min_conf
        ]

    @staticmethod
    def by_metadata(
        example: Example,
        key: str,
        value: Any,
        sources: Optional[List[EntitySource]] = None
    ) -> List[Entity]:
        """Filter entities by metadata field value.
        
        Searches for entities whose metadata dictionary contains the specified
        key-value pair. The value comparison uses == (equality check).
        
        Args:
            example: The Example object to search in
            key: Metadata key to search for
            value: Expected value for the metadata key
            sources: Optional list of EntitySource values to filter by.
                    If None, searches all sources.
                    
        Returns:
            List of entities with matching metadata
            
        Example:
            >>> example = Example(example_id="PMID_123", text="EGFR")
            >>> entity1 = Entity(
            ...     start_offset=0, end_offset=4, span_text="EGFR",
            ...     label="PROTEIN", source=EntitySource.GOLD,
            ...     metadata={"gene_id": "1956", "species": "human"}
            ... )
            >>> entity2 = Entity(
            ...     start_offset=0, end_offset=4, span_text="EGFR",
            ...     label="PROTEIN", source=EntitySource.GOLD,
            ...     metadata={"gene_id": "1956", "species": "mouse"}
            ... )
            >>> example.add_entities([entity1, entity2], EntitySource.GOLD)
            >>> human_entities = EntityQuery.by_metadata(example, "species", "human")
            >>> len(human_entities) == 1
            True
            >>> human_entities[0].metadata["species"] == "human"
            True
        """
        entities = example.get_all_entities(sources)
        return [
            e for e in entities
            if e.metadata is not None 
            and key in e.metadata 
            and e.metadata[key] == value
        ]

