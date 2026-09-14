from finetuning_lab.metrics import exact_match, normalized_edit_similarity, token_f1


def test_exact_match_normalizes_case_and_punctuation():
    assert exact_match("June 11, 1990", "june 11 1990") == 1.0


def test_token_f1_partial_overlap():
    score = token_f1("Ted Sanders", "Ted")
    assert 0.0 < score < 1.0


def test_edit_similarity_bounds():
    assert normalized_edit_similarity("abc", "abc") == 1.0
    assert 0.0 <= normalized_edit_similarity("abc", "xyz") <= 1.0
