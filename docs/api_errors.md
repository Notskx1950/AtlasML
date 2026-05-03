# API Error Contract

AtlasML uses explicit HTTP status codes for predictable API behavior.

## Common Errors

| Status | Meaning | Example |
|---:|---|---|
| 400 | Invalid input or missing artifact | Artifact path does not exist |
| 401 | Missing or invalid API key | Protected write endpoint without API key |
| 404 | Resource not found | Model version does not exist |
| 409 | Conflict | Duplicate model name/version |
| 500 | Unexpected server error | Unhandled inference failure |

## Registry Errors

### 400 Bad Request

Returned when the provided artifact URI is invalid.

```json
{
  "detail": "Artifact not found: /app/artifacts/model.joblib"
}