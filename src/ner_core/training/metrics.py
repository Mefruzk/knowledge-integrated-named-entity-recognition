from typing import List, Dict, Tuple
from collections import defaultdict
from numpy as np
from seqeval.metrics import classification_report
from seqeval.metrics import IOB2

def extract_entities(bio_tags: List[str]) -> List[Tuple[int, int, str]]:
    """
    Extract entities from BIO tag sequence

    Return: List of (start_idx, end_idx, entity_type) tuples
    """

    entities = []
    current_entity = None

    for idx, tag in enumerate(bio_tags):
        if tag.startswith('B-'):
            if current_entity:
                entities.append(current_entity)

            entity_type = tag[2:]
            current_entity = (idx, idx + 1, entity_type)

        elif tag.startswith('I-'):
            #Continue entity
            if current_entity:
                entity_type = tag[2:]
                if current_entity[2] == entity_type:
                    current_entity = (current_entity[0], idx + 1, entity_type)
                else:
                    entities.append(current_entity)
                    current_entity = (idx, idx + 1, entity_type)
        # 'O'
        else:
            # 'O' tag - end of current entity
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    
    if current_entity:
        entities.append(current_entity)

    return entities

def convert_to_bio_tags(
    predictions: List[List[int]],
    labels: List[List[int]],
    id2label: Dict[int, str]
) -> Tuple[List[List[str]], List[List[str]]]:

    # Convert predictions and labels to BIO tags, filtering -100 tags
    y_true, y_pred = [], []

    for pred_seq, label_seq in zip(predictions, labels):
        pred_tags = []
        true_tags = []
        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id != -100:
                pred_tags.append(id2label[pred_id])
                true_tags.append(id2label[label_id])

        y_pred.append(pred_tags)
        y_true.append(true_tags)

    return y_true, y_pred

def compute_entity_metrics(
    predictions: List[List[int]],
    labels: List[List[int]],
    id2label: Dict[int, str],
    return_report: bool = False
) -> Dict:

    """
    Compute entity-level NER metrics using seqeval
    """

    # Convert to BIO tags
    y_true, y_pred = convert_to_bio_tags(predictions, labels, id2label)

    # Get seqeval report
    report = classification_report(y_true, y_pred, output_dict=True, scheme=IOB2)

    # Flatten for W&B
    metrics = {}

    # Per-entity metrics
    for entity_type in ['PROTEIN', 'COMPLEX']:
        if entity_type in report:
            metrics[f'{entity_type}_precision'] = report[entity_type]['precision']
            metrics[f'{entity_type}_recall'] = report[entity_type]['recall']
            metrics[f'{entity_type}_f1'] = report[entity_type]['f1-score']

    # Overall metrics
    metrics['overall_precision'] = report['micro avg']['precision']
    metrics['overall_recall'] = report['micro avg']['recall']
    metrics['overall_f1'] = report['micro avg']['f1-score']

    if 'macro_avg' in report:
        metrics['macro_f1'] = report['macro avg']['f1-score']

    if return_report:
        return metrics, report

    return metrics

def compute_entity_confusion(
    predictions: List[List[int]],
    labels: List[List[int]],
    id2label: Dict[int, str]
) -> Dict:
    """
    Compute entity-level confusion matrix and boundary error analysis
    """
    
    # Initialize confusion counts
    confusion_counts = defaultdict(int)
    boundary_errors {'right_truncation': [], 'left_truncation': [], 'right_extension': [], 'left_extension': []}

    boundary_error_counts = defaultdict(lambda: defaultdict(int))

    # Convert to BIO tags
    y_true, y_pred = convert_to_bio_tags(predictions, labels, id2label)

    total_gold = 0
    total_pred = 0

    for true_tags, pred_tags in zip(y_true, y_pred):
        #Extract entities
        true_entities = extract_entities(true_tags)
        pred_entities = extract_entities(pred_tags)

        total_gold += len(true_entities)
        total_pred += len(pred_entities)

        # Fill Confusion Matrix #
        gold_by_span = {(start, end): ent_type for start, end, ent_type in true_entities}
        pred_by_span = {(start, end): ent_type for start, end, ent_type in pred_entities}

        processed_gold, processed_pred = set(), set()

        # Check predicted entities
        for pred_span, pred_type in pred_by_span.items():
            if pred_span in gold_by_span:
                #Exact span match
                gold_type = gold_by_span[pred_span]
                confusion_counts[f'{gold_type}->{pred_type}'] += 1
                processed_gold.add(pred_span)
                processed_pred.add(pred_span)
            else:
                # False positive
                confusion_counts[f'OTHER->{pred_type}'] += 1
                processed_pred.add(pred_span)

        # Check unmatched gold entities (missed)
        for gold_span, gold_type in gold_by_span.items():
            if gold_span not in processed_gold:
                confusion_counts[f'{gold_type}->OTHER'] += 1
                processed_gold.add(gold_span)

        
        # Boundary Error Analysis #
        # Check all gold-pred pairs for partial overlaps
        for gold_start, gold_end, gold_type in true_entities:
            for pred_start, pred_end, pred_type in pred_entities:

                # Skip exact matches
                if (gold_start, gold_end) == (pred_start, pred_end):
                    continue

                # Check for overlap
                overlap_start = max(gold_start, pred_start)
                overlap_end = min(gold_end, pred_end)

                if overlap_end <= overlap_start:
                    continue

                # Check if type error
                has_type_error = (gold_type != pred_type)

                #Partial overlap, log
                error_info = {
                    'gold_span' : (gold_start, gold_end),
                    'pred_span': (pred_start, pred_end),
                    'gold_type': gold_type,
                    'pred_type': pred_type,
                    'has_type_error': has_type_error,
                }

                # Case 1: Right truncation (pred missing right tokens)
                if pred_start == gold_start and pred_end < gold_end:
                    boundary_errors['right_truncation'].append(error_info)
                    boundary_error_counts[gold_type]['right_truncation'] += 1
                    if has_type_error:
                        boundary_error_counts[gold_type]['right_truncation_with_type_error'] += 1

                # Case 2: Left truncation (pred missing left tokens)
                if pred_start > gold_start and pred_end == gold_end:
                    boundary_errors['left_truncation'].append(error_info)
                    boundary_error_counts[gold_type]['left_truncation'] += 1
                    if has_type_error:
                        boundary_error_counts[gold_type]['left_truncation_with_type_error'] += 1
                
                # Case 3: Right extension (pred has extra right tokens)
                if pred_start == gold_start and pred_end > gold_end:
                    boundary_errors['right_extension'].append(error_info)
                    boundary_error_counts[gold_type]['right_extension'] += 1
                    if has_type_error:
                        boundary_error_counts[gold_type]['right_extension_with_type_error'] += 1
                
                # Case 4: Left extension (pred has extra left tokens)
                if pred_start < gold_start and pred_end == gold_end:
                    boundary_errors['left_extension'].append(error_info)
                    boundary_error_counts[gold_type]['left_extension'] += 1
                    if has_type_error:
                        boundary_error_counts[gold_type]['left_extension_with_type_error'] += 1
    
    #Compute summary#
    # True Positives: Exact span + type match
    exact_matches = sum(
        count for key, count in confusion_counts.items()
        if key.split('->')[0] == key.split('->')[1]
    )

    # Type errors: exact span, wrong type
    type_errors = sum(
        count for key, count in confusion_counts.items()
        if key.split('->')[0] not in ['OTHER'] and key.split('->')[1] not in ['OTHER']
        and if key.split('->')[0] != key.split('->')[1] 
    )

    # Missed entities (no prediction at gold span)
    missed_entities = sum(
        count for key, count in confusion_counts.items()
        if key.endswith('->OTHER')
    )

    # hallucinations (prediction where no gold entity)
    hallucinations = sum(
        count for key, count in confusion_counts.items()
        if key.startswith('OTHER->')
    )

    # False Negatives: Gold entities not correctly predicted
    # FN = Total gold - TP
    false_negatives = total_gold - exact_matches

    # False Positives: Predictions not correctly matching gold
    # FP - Total pred - TP
    false_positives = total_pred - exact_matches

    # Boundary errors total
    boundary_errors_total = sum(len(errors) for errors in boundary_errors.values())
    
    # Compute totals per entity type
    for entity_type in ['PROTEIN', 'COMPLEX']:
        total = sum(
            count for key, count in boundary_error_counts[entity_type].items()
            if not key.endswith('_with_type_error')
        )

        boundary_error_counts[entity_type]['total'] = total

    return {
    'confusion_matrix': dict(confusion_counts),
    'boundary_errors': boundary_errors,
    'boundary_error_counts': dict(boundary_error_counts),
    'summary': {
        'total_gold_entities': total_gold,
        'total_pred_entities': total_pred,
        'exact_matches': exact_matches,
        'type_errors': type_errors,
        'missed_entities': missed_entities,
        'hallucinations': hallucinations,
        'false_negatives': false_negatives,
        'false_positives': false_positives,
        'boundary_errors_total': boundary_errors_total,
    }
}

    