"""Actionable handling for Gemini API quota failures."""

GEMINI_QUOTA_URL = "https://ai.google.dev/gemini-api/docs/rate-limits"


def is_gemini_quota_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in ("429", "quota", "resource_exhausted", "rate limit")
    )


def explain_gemini_error(exc: Exception) -> str:
    if is_gemini_quota_error(exc):
        return (
            "Gemini API quota or rate limit reached for this key/project. Check Google AI "
            "Studio usage and rate limits, then wait for the limit to reset or enable "
            "billing. If ingestion was interrupted, rerun it without --reset."
        )
    return f"Could not complete the Gemini request: {exc}"
