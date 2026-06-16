# Add these methods to your OutputParser class to fix the validation errors

@staticmethod
def _extract_budget_range(questionnaire_summary: dict, original_answers: dict) -> str:
    """
    Extract budget_range from questionnaire_summary or original answers.
    Handles common variations.
    """
    # Try direct field first
    if "budget_range" in questionnaire_summary:
        val = questionnaire_summary.get("budget_range")
        if val and isinstance(val, str) and val.strip():
            return val
    
    # Try to extract from budget field in questionnaire_summary
    if "budget" in questionnaire_summary:
        val = questionnaire_summary.get("budget")
        if val and isinstance(val, str) and val.strip():
            return val
    
    # Try original answers
    if "budget" in original_answers:
        val = original_answers.get("budget")
        if val and isinstance(val, str) and val.strip():
            return val
    
    # Fallback
    return "Unknown"


@staticmethod
def _transform_optimisation_tips(tips_raw: Any) -> list[dict]:
    """
    Transform optimisation_tips from LLM format to schema format.
    LLM sometimes returns: [{"tip": "..."}]
    Schema expects: [{"impact": "...", "title": "...", "detail": "..."}]
    """
    if not isinstance(tips_raw, list):
        return []
    
    transformed = []
    for tip in tips_raw:
        if not isinstance(tip, dict):
            continue
        
        # If already has the correct structure, keep it
        if "impact" in tip and "title" in tip and "detail" in tip:
            transformed.append({
                "impact": tip.get("impact", "medium"),
                "title": tip.get("title", ""),
                "detail": tip.get("detail", ""),
            })
        # If has "tip" field, extract and transform
        elif "tip" in tip:
            tip_text = str(tip.get("tip", "")).strip()
            if tip_text:
                # Infer impact from content if possible, default to "medium"
                impact = "medium"
                if any(word in tip_text.lower() for word in ["critical", "significant", "major"]):
                    impact = "high"
                elif any(word in tip_text.lower() for word in ["minor", "optional", "consider"]):
                    impact = "low"
                
                transformed.append({
                    "impact": impact,
                    "title": tip_text[:80],  # Use first 80 chars as title
                    "detail": tip_text,
                })
        # Fallback: create placeholder
        else:
            transformed.append({
                "impact": "medium",
                "title": "Optimization Opportunity",
                "detail": "",
            })
    
    return transformed


@staticmethod
def _ensure_confidence_score(confidence: dict) -> dict:
    """
    Ensure confidence.score exists.
    If missing, infer from reason content or default to "medium".
    """
    if "score" not in confidence or not confidence.get("score"):
        # Try to infer from reason
        reason = str(confidence.get("reason", "")).lower()
        if any(word in reason for word in ["high confidence", "confident", "strong", "clearly"]):
            confidence["score"] = "high"
        elif any(word in reason for word in ["uncertain", "limited", "weak", "unclear"]):
            confidence["score"] = "low"
        else:
            confidence["score"] = "medium"
    
    return confidence


# Update the _normalize method to include these fixes
@staticmethod
def _normalize_enhanced(data: dict, input_hash: str, original_answers: dict = None) -> dict:
    """
    Enhanced normalization with all fixes.
    Pass original_answers (the raw questionnaire answers) for better extraction.
    """
    if original_answers is None:
        original_answers = {}
    
    # ========== EXISTING FIXES ==========
    # 1. Rename aliased top-level keys
    _KEY_ALIASES = {
        "workload_assumptions":         "workload_profile",
        "workload":                     "workload_profile",
        "recommendations":              "single_model_recommendations",
        "model_recommendations":        "single_model_recommendations",
        "optimization_opportunities":   "optimisation_tips",
        "optimization_tips":            "optimisation_tips",
    }
    
    for wrong, right in _KEY_ALIASES.items():
        if wrong in data and right not in data:
            data[right] = data.pop(wrong)

    # 2. Fix input_hash
    if data.get("input_hash") != input_hash:
        data["input_hash"] = input_hash

    # ========== NEW FIX #1: questionnaire_summary.budget_range ==========
    qs = data.get("questionnaire_summary", {})
    if isinstance(qs, dict):
        if "budget_range" not in qs or not qs.get("budget_range"):
            qs["budget_range"] = OutputParser._extract_budget_range(qs, original_answers)

    # 3. confidence field fixes
    confidence = data.get("confidence", {})
    if isinstance(confidence, dict):
        # rationale → reason
        if "rationale" in confidence and "reason" not in confidence:
            confidence["reason"] = confidence.pop("rationale")
        # assumptions must be a list
        if not isinstance(confidence.get("assumptions"), list):
            raw = confidence.get("assumptions", [])
            confidence["assumptions"] = [raw] if isinstance(raw, str) and raw else []
    
    # ========== NEW FIX #2: confidence.score ==========
    confidence = OutputParser._ensure_confidence_score(confidence)

    # 4. architecture defaults
    arch = data.get("architecture", {})
    if isinstance(arch, dict):
        arch.setdefault("hosting_strategy", "Managed API")
        arch.setdefault("pattern", "Single Model")
        arch.setdefault("framework_constraints", [])
        arch.setdefault("roles", [])
        arch.setdefault("notes", [])

    # 5. workload_profile Literal normalisations
    _latency_map = {
        "fast": "low", "very fast": "low", "real-time": "low",
        "low latency": "low",
        "moderate": "medium", "normal": "medium", "standard": "medium",
        "slow": "high", "batch": "high", "relaxed": "high",
    }
    _complexity_map = {
        "simple": "low", "easy": "low", "basic": "low",
        "moderate": "medium", "average": "medium", "medium complexity": "medium",
        "complex": "high", "difficult": "high", "very high": "high", "hard": "high",
    }
    workload = data.get("workload_profile", {})
    if isinstance(workload, dict):
        lat = str(workload.get("latency_requirement", "")).strip().lower()
        if lat and lat not in ("low", "medium", "high"):
            workload["latency_requirement"] = _latency_map.get(lat, "medium")
        cmp = str(workload.get("complexity", "")).strip().lower()
        if cmp and cmp not in ("low", "medium", "high"):
            workload["complexity"] = _complexity_map.get(cmp, "medium")

    # 6. single_model_recommendations — handle nested dict shape
    smr = data.get("single_model_recommendations")
    if isinstance(smr, dict):
        rebuilt = []
        for cat in ("recommended", "budget", "premium"):
            val = smr.get(cat)
            if isinstance(val, str):
                rebuilt.append({
                    "category": cat, "model_id": val,
                    "why": "", "tradeoffs": ""
                })
            elif isinstance(val, dict):
                val.setdefault("category", cat)
                rebuilt.append(val)
        data["single_model_recommendations"] = rebuilt

    # ========== NEW FIX #3: optimisation_tips structure ==========
    opt_tips = data.get("optimisation_tips", [])
    if isinstance(opt_tips, list):
        data["optimisation_tips"] = OutputParser._transform_optimisation_tips(opt_tips)

    # 7. questionnaire_summary — build from answers if missing
    if "questionnaire_summary" not in data:
        data["questionnaire_summary"] = {
            "app_type":     "Unknown",
            "agentic_level": "Non-Agentic",
            "scale":        "Unknown",
            "priority":     "Balanced",
            "budget_range": "Unknown",
        }

    return data
