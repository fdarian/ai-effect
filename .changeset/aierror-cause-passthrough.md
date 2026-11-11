---
"ai-effect": patch
---

`AiError.cause` is now the raw SDK error (the same value as `reason.cause`), so code that walks `.cause` chains finds `statusCode` and `responseBody` again.
