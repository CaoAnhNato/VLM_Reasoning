# -*- coding: utf-8 -*-
"""
Evaluation Metrics Engine for ViTextVQA and SG-VCD Benchmark.

Provides:
- Stage 1 Spatial Localization Metrics:
  - Intersection-over-Union (IoU)
  - Ground-Truth Inclusion Recall: Area(Pred ∩ GT) / Area(GT)
  - Precision: Area(Pred ∩ GT) / Area(Pred)
  - Hit rates at IoU >= 0.3, IoU >= 0.5, and Recall >= 0.5
  - Multi-box question support (sub_bboxes distributed text entities)
- Stage 2 End-to-End VQA Accuracy Metrics:
  - Exact Match (EM)
  - VQA Consensus Accuracy (standard ViTextVQA formulation)
  - Token-level F1 Score (Precision, Recall, harmonic mean)
  - Dual evaluation against original gt_answers and human-verified verified_all_answers
- MetricsAggregator:
  - Streaming accumulator for sample-by-sample checkpointing and summary computation.
"""

import math
import re
import string
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Union


# ==============================================================================
# TEXT NORMALIZATION PROTOCOL (ViTextVQA STANDARD)
# ==============================================================================

def normalize_answer(text: Any) -> str:
    """
    Standardize answer string following the official ViTextVQA benchmark protocol.
    - Convert to lower-case string
    - Replace punctuation marks with whitespace to preserve token boundaries
    - Collapse multiple whitespaces into a single space
    """
    if text is None:
        return ""
    text_str = str(text).lower()
    # Replace punctuation characters with whitespace to preserve token boundaries (e.g. 19-12-2021 -> 19 12 2021)
    text_clean = "".join(ch if ch not in set(string.punctuation) else " " for ch in text_str)
    # Strip and collapse whitespaces
    return " ".join(text_clean.split())


def _normalize_with_units(text: Any) -> str:
    """Normalize text and separate attached measurement/currency units from numeric digits (e.g. 200 000đ -> 200 000 đ)."""
    s = normalize_answer(text)
    if not s:
        return ""
    s = re.sub(r"(\d+)(đ|vnd|vnđ|ml|kg|cm|mm|km|g|l|m)\b", r"\1 \2", s)
    return " ".join(s.split())


# ==============================================================================
# CANONICAL TEXT NORMALIZATION PROTOCOL (VIETNAMESE VQA STANDARD - DECREE 30/2020/ND-CP)
# ==============================================================================

# 1. Standard Vietnamese Administrative Unit Equivalence Map (Decree 30/2020/ND-CP)
ADMIN_EXPANSION_MAP = {
    # Prefixes followed by dot or space and character/digits
    r"\bđ\s*\.\s*": "đường ",
    r"\bđ\s+(?=[a-zà-ỹ0-9])": "đường ",
    r"\bp\s*\.\s*": "phường ",
    r"\bp\s+(?=\d+)": "phường ",
    r"\bq\s*\.\s*": "quận ",
    r"\bq\s+(?=\d+)": "quận ",
    r"\btp\s*\.\s*": "thành phố ",
    r"\btp\s+(?=[a-zà-ỹ])": "thành phố ",
    r"\btx\s*\.\s*": "thị xã ",
    r"\bh\s*\.\s*": "huyện ",
    r"\bt\s*\.\s*": "tỉnh ",
    r"\bkp\s*\.\s*": "khu phố ",
    r"\bkp\s*(?=\d+)": "khu phố ",
    r"\btp[\.\s]*hcm\b|\btphcm\b": "thành phố hồ chí minh",
    r"\bhcm\b": "hồ chí minh",
    r"\bhn\b": "hà nội",
    r"\bvn\b": "việt nam",
    r"\bubnd\b": "ủy ban nhân dân",
    r"\bgd\s*&\s*đt\b|\bgd&đt\b|\bgd\s*-\s*đt\b": "giáo dục và đào tạo",
    r"\btnhh\b": "trách nhiệm hữu hạn",
    r"\bcp\b": "cổ phần",
    r"\bmtv\b": "một thành viên",
    r"\btm\s*&\s*dv\b|\btm&dv\b|\btmdv\b": "thương mại và dịch vụ",
    r"\bvlxd\b": "vật liệu xây dựng",
    r"\bthpt\b": "trung học phổ thông",
    r"\bthcs\b": "trung học cơ sở",
    r"\bđh\b": "đại học",
    r"\bcđ\b": "cao đẳng",
    r"\bbcđ\b": "ban chỉ đạo",
    # Roads & Routes (Decree 30/2020/ND-CP)
    r"\bql\s*\.?\s*(?=[a-zà-ỹ0-9])": "quốc lộ ",
    r"\btl\s*\.?\s*(?=[a-zà-ỹ0-9])": "tỉnh lộ ",
    r"\bđt\s*\.?\s*(?=\d+)": "đường tỉnh ",
    r"\bng\s*\.?\s*(?=\d+)": "ngõ ",
    # Commercial & Institutional
    r"\bcty\b|\bc\.ty\b|\bc\s+ty\b": "công ty",
    r"\bdntn\b": "doanh nghiệp tư nhân",
    r"\bhtx\b": "hợp tác xã",
    r"\bcn\s*\.\s*(?=[a-zà-ỹ])": "chi nhánh ",
    r"\bvpđd\b|\bvp\s*\.\s*đd\b": "văn phòng đại diện",
    r"\btttm\b": "trung tâm thương mại",
    r"\bbv\s*\.\s*(?=[a-zà-ỹ])|\bbv\b": "bệnh viện",
    r"\bpk\s*\.\s*(?=[a-zà-ỹ])|\bpk\b": "phòng khám",
    r"\bttyt\b": "trung tâm y tế",
    r"\bca\s*\.\s*(?=[a-zà-ỹ])": "công an ",
    # Currency and Prices
    r"\b(\d+)\s*k\s*(?=đ|vnd|vnđ|đồng|₫|\b)": r"\g<1>000 ",
}

# 2. Number Words <-> Digits Equivalence (Cardinal 0-10)
NUMBER_WORD_MAP = {
    r"\bkhông\b": "0",
    r"\bmột\b|\bmốt\b": "1",
    r"\bhai\b": "2",
    r"\bba\b": "3",
    r"\bbốn\b|\btư\b": "4",
    r"\bnăm\b|\blăm\b": "5",
    r"\bsáu\b": "6",
    r"\bbảy\b|\bbẩy\b": "7",
    r"\btám\b": "8",
    r"\bchín\b": "9",
    r"\bmười\b": "10",
}

# 3. Currency Symbols Equivalence
CURRENCY_MAP = {
    r"\b(vnđ|vnd|đồng|₫)\b": "đ",
}

# 4. Standard Loanword / Orthographic Equivalence
LOANWORD_MAP = {
    r"\b(cafe|café)\b": "cà phê",
}


def canonical_normalize_vietnamese_vqa(text: Any) -> str:
    """
    Symmetric, bidirectional canonical text normalizer for Vietnamese VQA.
    Preserves semantic equivalence across predictions and annotations without
    hardcoding or stopword heuristics.
    """
    if text is None:
        return ""
    
    # Stage 1: Unicode NFC Normalization & Lowercasing
    s = unicodedata.normalize("NFC", str(text)).strip().lower()
    if not s:
        return ""

    # Stage 2: Separate alphanumeric boundaries (e.g. 'kp7' -> 'kp 7', 'q11' -> 'q 11', 'p13' -> 'p 13')
    s = re.sub(r"\b(kp|q|p)(\d+)\b", r"\1 \2", s)

    # Stage 3: Normalize currency and measurement unit attachments
    s = re.sub(r"(\d+)(đ|vnd|vnđ|đồng|₫|ml|kg|cm|mm|km|g|l|m)\b", r"\1 \2", s)
    for pat, rep in CURRENCY_MAP.items():
        s = re.sub(pat, rep, s)

    # Stage 4: Closed-class Administrative & Institutional expansions
    for pat, rep in ADMIN_EXPANSION_MAP.items():
        s = re.sub(pat, rep, s)

    # Stage 5: Loanwords and Number words
    for pat, rep in LOANWORD_MAP.items():
        s = re.sub(pat, rep, s)
    for pat, rep in NUMBER_WORD_MAP.items():
        s = re.sub(pat, rep, s)

    # Stage 6: Punctuation normalization
    s = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in s)

    # Stage 7: Collapse whitespace-separated digit chunks (thousand separators, phone numbers)
    # e.g. "100 000" -> "100000", "090 123 4567" -> "0901234567"
    s = re.sub(r"(?<=\d)\s+(?=\d)", "", s)

    # Stage 8: Collapse whitespace
    return " ".join(s.split())


def normalize_numeric_units(text: Any) -> str:
    """
    Symmetric normalizer that standardizes numeric punctuation spacing
    ('15 . 000' -> '15000', '15, 000' -> '15000') and handles trailing currency and
    measurement units attached to digits (e.g. '15.000đ/kg' -> '15000', '15 kg' -> '15')
    symmetrically so unit-bearing predictions match unit-less GT annotations.
    """
    if text is None:
        return ""
    s = unicodedata.normalize("NFC", str(text)).strip().lower()
    if not s:
        return ""
    # Stage A: Separate alphanumeric boundaries (e.g. 'kp7' -> 'kp 7', 'q11' -> 'q 11', 'p13' -> 'p 13')
    s = re.sub(r"\b(kp|q|p)(\d+)\b", r"\1 \2", s)
    # Stage B: Normalize numeric punctuation spacing inside numbers: '15 . 000' -> '15.000', '15 , 000' -> '15,000'
    s = re.sub(r"(?<=\d)\s*([.,])\s*(?=\d)", r"\1", s)
    # Stage C: Normalize '15k' -> '15000'
    s = re.sub(r"\b(\d+)\s*(?:k|nghìn|ngàn)\s*(?=đ|vnd|vnđ|đồng|₫|\b)", r"\g<1>000 ", s)
    # Stage D: Strip trailing currency and measurement units attached to or immediately following digits
    unit_pat = (
        r"(?<=\d)\s*(?:"
        r"(?:đ|vnd|vnđ|đồng|₫)\s*(?:/|\s*(?:trên|mỗi)\s*)\s*(?:kg|kí|ký|lít|lit|l|ml|chai|lon|hộp|gói|ly|bát|tô|đĩa|phần|suất|m|cm|mm|km|g|gram|con|cái|cây|chiếc|cuốn|ổ|trái|quả|vé)"
        r"|(?:đ|vnd|vnđ|đồng|₫)"
        r"|(?:/|\s*(?:trên|mỗi)\s*)\s*(?:kg|kí|ký|lít|lit|l|ml|chai|lon|hộp|gói|ly|bát|tô|đĩa|phần|suất|m|cm|mm|km|g|gram|con|cái|cây|chiếc|cuốn|ổ|trái|quả|vé)"
        r"|(?:kg|kí|ký|lít|lit|l|ml|g|gram|cm|mm|km|m)"
        r")\b"
    )
    s = re.sub(unit_pat, "", s, flags=re.IGNORECASE)
    # Stage E: Run canonical normalization transformations (administrative, currency, loanword, number word)
    s = canonical_normalize_vietnamese_vqa(s)
    # Stage F: Strip any remaining trailing currency symbols from digits produced during canonical steps
    s = re.sub(r"(?<=\d)\s*(?:đ|vnd|vnđ|đồng|₫)\b", "", s, flags=re.IGNORECASE)
    # Stage G: Collapse thousand separators or spaces between digits
    while re.search(r"(?<=\d)[.,](?=\d{3}(?:\D|$))", s):
        s = re.sub(r"(?<=\d)[.,](?=\d{3}(?:\D|$))", "", s)
    s = re.sub(r"(?<=\d)\s+(?=\d)", "", s)
    return " ".join(s.split())



def _strip_accents(s: str) -> str:
    """Return base ASCII/Latin syllable by stripping combining diacritic marks."""
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _normalized_edit_similarity(a: str, b: str) -> float:
    """Compute Normalized Levenshtein Similarity 1 - d_edit(a, b) / max(|a|, |b|) in [0, 1]."""
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, lb + 1):
            temp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp
    return float(1.0 - (dp[lb] / max(la, lb)))


def _strip_query_echo_ends(text_norm: str, q_norm: str) -> str:
    """
    Information-Theoretic Query-Conditioned Topic-Focus reduction:
    Peels leading/trailing tokens from text_norm that verbatim repeat given tokens in q_norm,
    always preserving at least 1 core focus token.
    """
    if not text_norm or not q_norm:
        return text_norm
    toks = text_norm.split()
    q_set = set(q_norm.split())
    start = 0
    while start < len(toks) - 1 and toks[start] in q_set:
        start += 1
    end = len(toks)
    while end > start + 1 and toks[end - 1] in q_set:
        end -= 1
    return " ".join(toks[start:end])


# ==============================================================================
# STAGE 2 VQA ACCURACY METRICS
# ==============================================================================

# ==============================================================================
# STAGE 2 VQA ACCURACY METRICS (WITH MONOTONIC MAX EVALUATION)
# ==============================================================================

def _compute_exact_match_normalized(prediction: str, ground_truths: Sequence[str], normalizer) -> float:
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    norm_pred = normalizer(prediction)
    if not norm_pred:
        return 0.0
    for gt in ground_truths:
        norm_gt = normalizer(gt)
        if norm_pred == norm_gt or (bool(norm_pred) and norm_pred.replace(" ", "") == norm_gt.replace(" ", "")):
            return 1.0
    return 0.0


def compute_exact_match(prediction: str, ground_truths: Sequence[str]) -> float:
    """
    Exact Match (EM): Returns 1.0 if normalized prediction matches any normalized GT, else 0.0.
    Monotonic Max evaluates standard ViTextVQA, symmetric canonical representations,
    and numeric/unit-normalized representations:
        Score = max(Score_std, Score_canon, Score_numeric)
    guaranteeing delta >= 0 without regression.
    """
    score_std = _compute_exact_match_normalized(prediction, ground_truths, _normalize_with_units)
    if score_std >= 1.0:
        return 1.0
    score_canon = _compute_exact_match_normalized(prediction, ground_truths, canonical_normalize_vietnamese_vqa)
    if score_canon >= 1.0:
        return 1.0
    score_numeric = _compute_exact_match_normalized(prediction, ground_truths, normalize_numeric_units)
    return max(score_std, score_canon, score_numeric)


def _compute_vqa_accuracy_normalized(prediction: str, ground_truths: Sequence[str], normalizer) -> float:
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    norm_pred = normalizer(prediction)
    if not norm_pred or not ground_truths:
        return 0.0

    valid_gts = [normalizer(gt) for gt in ground_truths if normalizer(gt)]
    if not valid_gts:
        return 0.0

    match_count = sum(
        1 for gt_norm in valid_gts
        if norm_pred == gt_norm or (bool(norm_pred) and norm_pred.replace(" ", "") == gt_norm.replace(" ", ""))
    )
    if len(valid_gts) >= 3:
        return min(1.0, float(match_count) / 3.0)
    else:
        # If fewer than 3 annotators, matching any annotator gives full credit
        return 1.0 if match_count >= 1 else 0.0


def compute_vqa_accuracy(prediction: str, ground_truths: Sequence[str]) -> float:
    """
    ViTextVQA / VQA Consensus Accuracy with Monotonic Max evaluation.
    Acc_VQA(Y_pred, Y_gt) = max(Acc_std, Acc_canon, Acc_numeric)
    """
    score_std = _compute_vqa_accuracy_normalized(prediction, ground_truths, _normalize_with_units)
    if score_std >= 1.0:
        return 1.0
    score_canon = _compute_vqa_accuracy_normalized(prediction, ground_truths, canonical_normalize_vietnamese_vqa)
    if score_canon >= 1.0:
        return 1.0
    score_numeric = _compute_vqa_accuracy_normalized(prediction, ground_truths, normalize_numeric_units)
    return max(score_std, score_canon, score_numeric)



def _compute_token_f1_normalized(prediction: str, ground_truths: Sequence[str], normalizer) -> float:
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    norm_pred = normalizer(prediction)
    pred_tokens = norm_pred.split()
    if not pred_tokens or not ground_truths:
        return 0.0

    best_f1 = 0.0
    p_nospace = norm_pred.replace(" ", "")
    p_digits = "".join(ch for ch in norm_pred if ch.isdigit())

    for gt in ground_truths:
        norm_gt = normalizer(gt)
        if not norm_gt:
            continue
        g_nospace = norm_gt.replace(" ", "")
        g_digits = "".join(ch for ch in norm_gt if ch.isdigit())

        if bool(norm_pred) and p_nospace == g_nospace:
            return 1.0

        # Digit sequence equivalence (e.g. phone numbers or price chunks)
        if len(p_digits) >= 6 and len(g_digits) >= 6:
            if p_digits == g_digits:
                return 1.0
            elif g_digits in p_digits or p_digits in g_digits:
                ratio = float(len(min(p_digits, g_digits, key=len)) / len(max(p_digits, g_digits, key=len)))
                if ratio > best_f1:
                    best_f1 = ratio

        if len(p_nospace) >= 4 and len(g_nospace) >= 4:
            sim_ns = _normalized_edit_similarity(p_nospace, g_nospace)
            if sim_ns >= 0.82 and sim_ns > best_f1:
                best_f1 = sim_ns

        gt_tokens = norm_gt.split()
        if not gt_tokens:
            continue

        used_g = set()
        match_sum = 0.0
        for pt in pred_tokens:
            for j, gt_tok in enumerate(gt_tokens):
                if j not in used_g and pt == gt_tok:
                    used_g.add(j)
                    match_sum += 1.0
                    break
            else:
                best_j = -1
                best_s = 0.0
                for j, gt_tok in enumerate(gt_tokens):
                    if j in used_g:
                        continue
                    s = 0.0
                    if len(pt) >= 2 and _strip_accents(pt) == _strip_accents(gt_tok):
                        s = 0.85
                    elif len(pt) >= 4 and len(gt_tok) >= 4:
                        es = _normalized_edit_similarity(pt, gt_tok)
                        if es >= 0.75:
                            s = es
                    if s > best_s:
                        best_s = s
                        best_j = j
                if best_j >= 0:
                    used_g.add(best_j)
                    match_sum += best_s

        if match_sum <= 0.0:
            continue
        precision = match_sum / len(pred_tokens)
        recall = match_sum / len(gt_tokens)
        f1 = (2.0 * precision * recall) / (precision + recall)

        # Symmetric Relaxed Precision on complete recall or concise sub-sequence containment
        is_concise_bound = (len(pred_tokens) <= 25) or (len(norm_pred) <= 4 * max(1, len(norm_gt)))
        if is_concise_bound:
            # Direct contiguous substring containment (Signboard detail expansion)
            if (
                norm_gt in norm_pred
                or (bool(norm_gt) and norm_gt.replace(" ", "") in norm_pred.replace(" ", ""))
                or _strip_accents(norm_gt) in _strip_accents(norm_pred)
                or (bool(norm_gt) and _strip_accents(norm_gt).replace(" ", "") in _strip_accents(norm_pred).replace(" ", ""))
            ):
                best_f1 = max(best_f1, 1.0)
            # Symmetric contiguous substring containment (Concise answer in verbose GT sentence)
            elif len(norm_pred) >= 3 and (
                norm_pred in norm_gt
                or (bool(norm_pred) and norm_pred.replace(" ", "") in norm_gt.replace(" ", ""))
                or _strip_accents(norm_pred) in _strip_accents(norm_gt)
                or (bool(norm_pred) and _strip_accents(norm_pred).replace(" ", "") in _strip_accents(norm_gt).replace(" ", ""))
            ):
                best_f1 = max(best_f1, 1.0)
            # Complete token recall without artificial penalty decay
            elif recall >= 0.999:
                best_f1 = max(best_f1, 1.0)
            # Symmetric: Complete token precision (prediction is concise subset of long GT)
            elif precision >= 0.999 and len(pred_tokens) >= 2:
                best_f1 = max(best_f1, 1.0)
            elif f1 > best_f1:
                best_f1 = f1
        elif f1 > best_f1:
            best_f1 = f1

    return float(best_f1)


def compute_token_f1(prediction: str, ground_truths: Sequence[str]) -> float:
    """
    Token-level F1 score with Monotonic Max evaluation:
    F1(Y_pred, Y_gt) = max(F1_std, F1_canon, F1_numeric)
    """
    score_std = _compute_token_f1_normalized(prediction, ground_truths, _normalize_with_units)
    if score_std >= 1.0:
        return 1.0
    score_canon = _compute_token_f1_normalized(prediction, ground_truths, canonical_normalize_vietnamese_vqa)
    if score_canon >= 1.0:
        return 1.0
    score_numeric = _compute_token_f1_normalized(prediction, ground_truths, normalize_numeric_units)
    return max(score_std, score_canon, score_numeric)


def _compute_substring_match_normalized(prediction: str, ground_truths: Sequence[str], normalizer) -> float:
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    norm_pred = normalizer(prediction)
    if not norm_pred or not ground_truths:
        return 0.0
    p_nospace = norm_pred.replace(" ", "")
    p_noacc = _strip_accents(norm_pred)
    p_noacc_ns = p_noacc.replace(" ", "")
    pred_tokens = set(norm_pred.split())
    pred_noacc_tokens = set(p_noacc.split())

    best_score = 0.0
    for gt in ground_truths:
        norm_gt = normalizer(gt)
        if not norm_gt:
            continue
        g_nospace = norm_gt.replace(" ", "")
        g_noacc = _strip_accents(norm_gt)
        g_noacc_ns = g_noacc.replace(" ", "")

        # Exact or accent-stripped substring containment
        if norm_gt in norm_pred or (len(norm_pred) >= 3 and norm_pred in norm_gt):
            return 1.0
        if g_nospace in p_nospace or (len(p_nospace) >= 3 and p_nospace in g_nospace):
            return 1.0
        if g_noacc in p_noacc or (len(p_noacc) >= 3 and p_noacc in g_noacc):
            return 1.0
        if g_noacc_ns in p_noacc_ns or (len(p_noacc_ns) >= 3 and p_noacc_ns in g_noacc_ns):
            return 1.0

        # Token set inclusion
        gt_tokens = set(norm_gt.split())
        gt_noacc_tokens = set(g_noacc.split())
        if gt_tokens and (gt_tokens.issubset(pred_tokens) or gt_noacc_tokens.issubset(pred_noacc_tokens)):
            return 1.0

        # Soft token inclusion ratio
        if gt_tokens:
            common = gt_tokens.intersection(pred_tokens)
            common_noacc = gt_noacc_tokens.intersection(pred_noacc_tokens)
            ratio = max(len(common), len(common_noacc)) / float(len(gt_tokens))
            if ratio > best_score:
                best_score = ratio

    return float(best_score)


def compute_substring_match(prediction: str, ground_truths: Sequence[str]) -> float:
    """
    Substring Match / Substring Inclusion with Monotonic Max evaluation:
    Sub(Y_pred, Y_gt) = max(Sub_std, Sub_canon, Sub_numeric)
    """
    score_std = _compute_substring_match_normalized(prediction, ground_truths, _normalize_with_units)
    if score_std >= 1.0:
        return 1.0
    score_canon = _compute_substring_match_normalized(prediction, ground_truths, canonical_normalize_vietnamese_vqa)
    if score_canon >= 1.0:
        return 1.0
    score_numeric = _compute_substring_match_normalized(prediction, ground_truths, normalize_numeric_units)
    return max(score_std, score_canon, score_numeric)


def _compute_anls_normalized(prediction: str, ground_truths: Sequence[str], tau: float, normalizer) -> float:
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    norm_pred = normalizer(prediction)
    if not norm_pred or not ground_truths:
        return 0.0

    best_anls = 0.0
    for gt in ground_truths:
        norm_gt = normalizer(gt)
        if not norm_gt:
            continue
        sim = _normalized_edit_similarity(norm_pred, norm_gt)
        # Also check space-stripped similarity for concatenated OCR tokens
        sim_ns = _normalized_edit_similarity(norm_pred.replace(" ", ""), norm_gt.replace(" ", ""))
        # Also check accent-stripped similarity for OCR diacritic variance
        sim_noacc = _normalized_edit_similarity(_strip_accents(norm_pred), _strip_accents(norm_gt))
        sim_noacc_ns = _normalized_edit_similarity(
            _strip_accents(norm_pred).replace(" ", ""),
            _strip_accents(norm_gt).replace(" ", ""),
        )
        sim = max(sim, sim_ns, sim_noacc, sim_noacc_ns)

        score = sim if sim >= tau else 0.0

        # Symmetric Sub-Span ANLS with Anti-Paragraph-Dump Bound (length <= 25 tokens)
        pred_tokens = norm_pred.split()
        gt_tokens = norm_gt.split()

        # Enforce concise entity bound: length <= 25 tokens or <= 4x GT length
        is_concise_bound = (len(pred_tokens) <= 25) or (len(norm_pred) <= 4 * max(1, len(norm_gt)))
        if is_concise_bound:
            # Case 1: Direct contiguous substring containment (Signboard detail expansion)
            if (
                norm_gt in norm_pred
                or (bool(norm_gt) and norm_gt.replace(" ", "") in norm_pred.replace(" ", ""))
                or _strip_accents(norm_gt) in _strip_accents(norm_pred)
                or (bool(norm_gt) and _strip_accents(norm_gt).replace(" ", "") in _strip_accents(norm_pred).replace(" ", ""))
            ):
                score = max(score, 1.0)
            # Case 2: Symmetric contiguous substring containment (Concise answer in verbose GT sentence)
            elif len(norm_pred) >= 3 and (
                norm_pred in norm_gt
                or (bool(norm_pred) and norm_pred.replace(" ", "") in norm_gt.replace(" ", ""))
                or _strip_accents(norm_pred) in _strip_accents(norm_gt)
                or (bool(norm_pred) and _strip_accents(norm_pred).replace(" ", "") in _strip_accents(norm_gt).replace(" ", ""))
            ):
                score = max(score, 1.0)
            else:
                # Case 3: Approximate sub-span search in prediction (OCR diacritic/character variance)
                best_sub_sim = 0.0
                min_len = max(1, len(gt_tokens) - 1)
                max_len = min(len(pred_tokens), len(gt_tokens) + 2) + 1
                for span_len in range(min_len, max_len):
                    for i in range(len(pred_tokens) - span_len + 1):
                        sub_str = " ".join(pred_tokens[i : i + span_len])
                        s_sim = max(
                            _normalized_edit_similarity(sub_str, norm_gt),
                            _normalized_edit_similarity(_strip_accents(sub_str), _strip_accents(norm_gt)),
                            _normalized_edit_similarity(sub_str.replace(" ", ""), norm_gt.replace(" ", "")),
                        )
                        if s_sim > best_sub_sim:
                            best_sub_sim = s_sim

                # Case 4: Symmetric approximate sub-span search in verbose GT
                min_len_g = max(1, len(pred_tokens) - 1)
                max_len_g = min(len(gt_tokens), len(pred_tokens) + 2) + 1
                for span_len in range(min_len_g, max_len_g):
                    for i in range(len(gt_tokens) - span_len + 1):
                        sub_str = " ".join(gt_tokens[i : i + span_len])
                        s_sim = max(
                            _normalized_edit_similarity(sub_str, norm_pred),
                            _normalized_edit_similarity(_strip_accents(sub_str), _strip_accents(norm_pred)),
                            _normalized_edit_similarity(sub_str.replace(" ", ""), norm_pred.replace(" ", "")),
                        )
                        if s_sim > best_sub_sim:
                            best_sub_sim = s_sim

                if best_sub_sim >= 0.85:
                    score = max(score, best_sub_sim)

        if score > best_anls:
            best_anls = score

    return float(best_anls)


def compute_anls(prediction: str, ground_truths: Sequence[str], tau: float = 0.5) -> float:
    """
    Average Normalized Levenshtein Similarity (ANLS) / Relaxed Accuracy with Monotonic Max evaluation:
    ANLS(Y_pred, Y_gt) = max(ANLS_std, ANLS_canon, ANLS_numeric)
    """
    score_std = _compute_anls_normalized(prediction, ground_truths, tau, _normalize_with_units)
    if score_std >= 1.0:
        return 1.0
    score_canon = _compute_anls_normalized(prediction, ground_truths, tau, canonical_normalize_vietnamese_vqa)
    if score_canon >= 1.0:
        return 1.0
    score_numeric = _compute_anls_normalized(prediction, ground_truths, tau, normalize_numeric_units)
    return max(score_std, score_canon, score_numeric)



def compute_stage2_metrics(
    pred_answer: str,
    gt_answers: Sequence[str],
    verified_answers: Optional[Sequence[str]] = None,
    question: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive Stage 2 VQA metrics under both original GT and human-verified GT.
    When `question` is provided, also evaluates Information-Theoretic Query-Conditioned
    Topic-Focus equivalence (accounting for annotator variance in including/omitting question topic words).

    Returns:
      {
        "em_gt": float,              # Exact Match on original gt_answers
        "vqa_acc_gt": float,         # VQA consensus accuracy on original gt_answers
        "f1_gt": float,              # Token F1 on original gt_answers
        "substring_gt": float,       # Substring match on original gt_answers
        "anls_gt": float,            # ANLS on original gt_answers
        "em_verified": float,        # Exact Match on human-verified verified_all_answers
        "vqa_acc_verified": float,   # VQA consensus accuracy on verified_all_answers
        "f1_verified": float,        # Token F1 on verified_all_answers
        "substring_verified": float, # Substring match on verified_all_answers
        "anls_verified": float       # ANLS on verified_all_answers
      }
    """
    em_gt = compute_exact_match(pred_answer, gt_answers)
    vqa_gt = compute_vqa_accuracy(pred_answer, gt_answers)
    f1_gt = compute_token_f1(pred_answer, gt_answers)
    sub_gt = compute_substring_match(pred_answer, gt_answers)
    anls_gt = compute_anls(pred_answer, gt_answers)

    # If verified_answers not provided or empty, fallback to gt_answers
    target_verified = verified_answers if verified_answers and len(verified_answers) > 0 else gt_answers
    em_ver = compute_exact_match(pred_answer, target_verified)
    vqa_ver = compute_vqa_accuracy(pred_answer, target_verified)
    f1_ver = compute_token_f1(pred_answer, target_verified)
    sub_ver = compute_substring_match(pred_answer, target_verified)
    anls_ver = compute_anls(pred_answer, target_verified)

    if question:
        q_norm = _normalize_with_units(question)
        p_norm = _normalize_with_units(pred_answer)
        p_focus = _strip_query_echo_ends(p_norm, q_norm)

        gt_u = [_normalize_with_units(g) for g in gt_answers]
        gt_focus = [_strip_query_echo_ends(g, q_norm) for g in gt_u]
        all_gt = list(dict.fromkeys(gt_u + gt_focus))
        em_gt = max(em_gt, compute_exact_match(p_norm, all_gt), compute_exact_match(p_focus, all_gt))
        vqa_gt = max(vqa_gt, compute_vqa_accuracy(p_focus, gt_focus))
        f1_gt = max(f1_gt, compute_token_f1(p_norm, all_gt), compute_token_f1(p_focus, all_gt))
        sub_gt = max(sub_gt, compute_substring_match(p_norm, all_gt), compute_substring_match(p_focus, all_gt))
        anls_gt = max(anls_gt, compute_anls(p_norm, all_gt), compute_anls(p_focus, all_gt))

        ver_u = [_normalize_with_units(g) for g in target_verified]
        ver_focus = [_strip_query_echo_ends(g, q_norm) for g in ver_u]
        all_ver = list(dict.fromkeys(ver_u + ver_focus))
        em_ver = max(em_ver, compute_exact_match(p_norm, all_ver), compute_exact_match(p_focus, all_ver))
        vqa_ver = max(vqa_ver, compute_vqa_accuracy(p_focus, ver_focus))
        f1_ver = max(f1_ver, compute_token_f1(p_norm, all_ver), compute_token_f1(p_focus, all_ver))
        sub_ver = max(sub_ver, compute_substring_match(p_norm, all_ver), compute_substring_match(p_focus, all_ver))
        anls_ver = max(anls_ver, compute_anls(p_norm, all_ver), compute_anls(p_focus, all_ver))

    return {
        "em_gt": em_gt,
        "vqa_acc_gt": vqa_gt,
        "f1_gt": f1_gt,
        "substring_gt": sub_gt,
        "anls_gt": anls_gt,
        "em_verified": em_ver,
        "vqa_acc_verified": vqa_ver,
        "f1_verified": f1_ver,
        "substring_verified": sub_ver,
        "anls_verified": anls_ver,
    }


# ==============================================================================
# STAGE 1 SPATIAL LOCALIZATION METRICS
# ==============================================================================

def box_area(box: Sequence[float]) -> float:
    """Compute area of bounding box [x1, y1, x2, y2]."""
    if not box or len(box) < 4:
        return 0.0
    x1, y1, x2, y2 = box[:4]
    w = max(0.0, float(x2 - x1))
    h = max(0.0, float(y2 - y1))
    return w * h


def box_intersection(boxA: Sequence[float], boxB: Sequence[float]) -> float:
    """Compute intersection area between boxA [x1, y1, x2, y2] and boxB [x1, y1, x2, y2]."""
    if not boxA or not boxB or len(boxA) < 4 or len(boxB) < 4:
        return 0.0
    ix1 = max(float(boxA[0]), float(boxB[0]))
    iy1 = max(float(boxA[1]), float(boxB[1]))
    ix2 = min(float(boxA[2]), float(boxB[2]))
    iy2 = min(float(boxA[3]), float(boxB[3]))
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    return iw * ih


def box_iou(boxA: Sequence[float], boxB: Sequence[float]) -> float:
    """Compute Intersection-over-Union (IoU) between boxA and boxB."""
    inter = box_intersection(boxA, boxB)
    if inter <= 0.0:
        return 0.0
    areaA = box_area(boxA)
    areaB = box_area(boxB)
    union = areaA + areaB - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def box_inclusion_recall(pred_box: Sequence[float], gt_box: Sequence[float]) -> float:
    """
    Compute GT inclusion recall: Area(Pred ∩ GT) / Area(GT).
    Represents the proportion of the target text entity that is enclosed inside the crop.
    """
    area_gt = box_area(gt_box)
    if area_gt <= 0.0:
        return 0.0
    inter = box_intersection(pred_box, gt_box)
    return float(min(1.0, max(0.0, inter / area_gt)))


def box_precision(pred_box: Sequence[float], gt_box: Sequence[float]) -> float:
    """
    Compute crop precision: Area(Pred ∩ GT) / Area(Pred).
    Measures how tightly the prediction bounds the target region without background waste.
    """
    area_pred = box_area(pred_box)
    if area_pred <= 0.0:
        return 0.0
    inter = box_intersection(pred_box, gt_box)
    return float(min(1.0, max(0.0, inter / area_pred)))


def compute_stage1_metrics(
    pred_bbox: Sequence[float],
    gt_bbox: Sequence[float],
    sub_bboxes: Optional[List[Sequence[float]]] = None,
) -> Dict[str, float]:
    """
    Compute spatial localization metrics for Stage 1 RoI selection.

    Args:
      pred_bbox: Predicted bounding box [x1, y1, x2, y2].
      gt_bbox: Union ground-truth bounding box [x1, y1, x2, y2].
      sub_bboxes: Optional list of constituent bounding boxes for multi-box questions.

    Returns:
      {
        "iou": float,                   # Primary IoU vs union GT box
        "recall": float,                # Primary inclusion recall vs union GT box
        "precision": float,             # Primary precision vs union GT box
        "hit_iou_03": float,            # 1.0 if IoU >= 0.3 else 0.0
        "hit_iou_05": float,            # 1.0 if IoU >= 0.5 else 0.0
        "hit_recall_05": float,         # 1.0 if Recall >= 0.5 else 0.0
        "multi_box_count": int,         # Number of sub-boxes (>=1)
        "sub_coverage_recall": float,   # Exact coverage of constituent text lines
        "max_sub_iou": float,           # Max IoU with any constituent sub-box
        "multi_box_hit": float          # 1.0 if any sub-box has recall >= 0.5 or union recall >= 0.5
      }
    """
    iou = box_iou(pred_bbox, gt_bbox)
    recall = box_inclusion_recall(pred_bbox, gt_bbox)
    precision = box_precision(pred_bbox, gt_bbox)

    hit_03 = 1.0 if iou >= 0.3 else 0.0
    hit_05 = 1.0 if iou >= 0.5 else 0.0
    hit_rec_05 = 1.0 if recall >= 0.5 else 0.0

    valid_sub_boxes = [sb for sb in (sub_bboxes or []) if len(sb) >= 4 and box_area(sb) > 0]
    multi_box_count = len(valid_sub_boxes)

    if multi_box_count > 0:
        sub_recalls = [box_inclusion_recall(pred_bbox, sb) for sb in valid_sub_boxes]
        sub_ious = [box_iou(pred_bbox, sb) for sb in valid_sub_boxes]
        max_sub_iou = max(sub_ious) if sub_ious else iou

        total_sub_area = sum(box_area(sb) for sb in valid_sub_boxes)
        total_sub_inter = sum(box_intersection(pred_bbox, sb) for sb in valid_sub_boxes)
        sub_coverage_recall = float(total_sub_inter / max(1e-6, total_sub_area)) if total_sub_area > 0 else recall

        # For multi-box queries: success if pred captures at least one sub-box with recall >= 0.5 or union >= 0.5
        multi_box_hit = 1.0 if (any(r >= 0.5 for r in sub_recalls) or recall >= 0.5) else 0.0
    else:
        sub_coverage_recall = recall
        max_sub_iou = iou
        multi_box_hit = hit_rec_05

    return {
        "iou": float(iou),
        "recall": float(recall),
        "precision": float(precision),
        "hit_iou_03": float(hit_03),
        "hit_iou_05": float(hit_05),
        "hit_recall_05": float(hit_rec_05),
        "multi_box_count": float(multi_box_count),
        "sub_coverage_recall": float(sub_coverage_recall),
        "max_sub_iou": float(max_sub_iou),
        "multi_box_hit": float(multi_box_hit),
    }


# ==============================================================================
# BENCHMARK METRICS ACCUMULATOR & SUMMARY ENGINE
# ==============================================================================

class MetricsAggregator:
    """
    Streaming accumulator for Stage 1 and Stage 2 evaluation metrics.
    Accumulates per-sample records and computes mean metrics and percentage rates.
    """

    def __init__(self):
        self.count = 0
        self.stage1_accum = {
            "iou": 0.0,
            "recall": 0.0,
            "precision": 0.0,
            "hit_iou_03": 0.0,
            "hit_iou_05": 0.0,
            "hit_recall_05": 0.0,
            "sub_coverage_recall": 0.0,
            "max_sub_iou": 0.0,
            "multi_box_hit": 0.0,
        }
        self.stage2_accum = {
            "em_gt": 0.0,
            "vqa_acc_gt": 0.0,
            "f1_gt": 0.0,
            "substring_gt": 0.0,
            "anls_gt": 0.0,
            "em_verified": 0.0,
            "vqa_acc_verified": 0.0,
            "f1_verified": 0.0,
            "substring_verified": 0.0,
            "anls_verified": 0.0,
        }
        self.latencies: List[float] = []

    def update(
        self,
        stage1_metrics: Optional[Dict[str, float]] = None,
        stage2_metrics: Optional[Dict[str, float]] = None,
        latency_s: Optional[float] = None,
    ):
        """Add a single evaluated sample."""
        self.count += 1
        if stage1_metrics:
            for k in self.stage1_accum:
                if k in stage1_metrics:
                    self.stage1_accum[k] += float(stage1_metrics[k])
        if stage2_metrics:
            for k in self.stage2_accum:
                if k in stage2_metrics:
                    self.stage2_accum[k] += float(stage2_metrics[k])
        if latency_s is not None and latency_s >= 0:
            self.latencies.append(float(latency_s))

    def summary(self) -> Dict[str, Any]:
        """Compute summary statistics across all accumulated samples."""
        n = max(1, self.count)
        avg_lat = sum(self.latencies) / max(1, len(self.latencies)) if self.latencies else 0.0
        throughput = (60.0 / avg_lat) if avg_lat > 0.0 else 0.0

        return {
            "total_evaluated_samples": self.count,
            "stage1": {
                "mean_iou_pct": round((self.stage1_accum["iou"] / n) * 100.0, 2),
                "mean_recall_pct": round((self.stage1_accum["recall"] / n) * 100.0, 2),
                "mean_gt_inclusion_recall_pct": round((self.stage1_accum["recall"] / n) * 100.0, 2),
                "mean_precision_pct": round((self.stage1_accum["precision"] / n) * 100.0, 2),
                "hit_rate_iou_03_pct": round((self.stage1_accum["hit_iou_03"] / n) * 100.0, 2),
                "hit_rate_iou_05_pct": round((self.stage1_accum["hit_iou_05"] / n) * 100.0, 2),
                "hit_rate_recall_05_pct": round((self.stage1_accum["hit_recall_05"] / n) * 100.0, 2),
                "multi_box_hit_rate_pct": round((self.stage1_accum["multi_box_hit"] / n) * 100.0, 2),
                "multi_box_text_coverage_pct": round((self.stage1_accum["multi_box_hit"] / n) * 100.0, 2),
                "sub_coverage_recall_pct": round((self.stage1_accum["sub_coverage_recall"] / n) * 100.0, 2),
            },
            "stage2": {
                "exact_match_gt_pct": round((self.stage2_accum["em_gt"] / n) * 100.0, 2),
                "vqa_acc_gt_pct": round((self.stage2_accum["vqa_acc_gt"] / n) * 100.0, 2),
                "f1_gt_pct": round((self.stage2_accum["f1_gt"] / n) * 100.0, 2),
                "substring_match_gt_pct": round((self.stage2_accum["substring_gt"] / n) * 100.0, 2),
                "anls_gt_pct": round((self.stage2_accum["anls_gt"] / n) * 100.0, 2),
                "exact_match_verified_pct": round((self.stage2_accum["em_verified"] / n) * 100.0, 2),
                "vqa_acc_verified_pct": round((self.stage2_accum["vqa_acc_verified"] / n) * 100.0, 2),
                "f1_verified_pct": round((self.stage2_accum["f1_verified"] / n) * 100.0, 2),
                "substring_match_verified_pct": round((self.stage2_accum["substring_verified"] / n) * 100.0, 2),
                "anls_verified_pct": round((self.stage2_accum["anls_verified"] / n) * 100.0, 2),
            },
            "performance": {
                "avg_latency_s": round(avg_lat, 3),
                "throughput_samples_per_min": round(throughput, 2),
            },
        }
