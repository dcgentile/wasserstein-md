def calculate_f1_score_with_threshold(detected_points, true_values):
    true_positives = 0
    matched_true_positives = set()

    for detected_point in detected_points:
        for true_point in true_values:
            if true_point not in matched_true_positives and abs(detected_point - true_point) <= 100:
                true_positives += 1
                matched_true_positives.add(true_point)
                break 

    if (true_positives>len(true_values)):
        true_positives=len(true_values)
    false_positives = len(detected_points) - true_positives
    if (len(true_values) <= true_positives):
        false_negatives = 0
    else:
        false_negatives = len(true_values) - true_positives

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0

    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return f1_score, precision, recall