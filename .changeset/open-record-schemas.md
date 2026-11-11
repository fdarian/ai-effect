---
"ai-effect": patch
---

Free-form records like `Schema.Record(Schema.String, Schema.Unknown)` stay open in the tool and output JSON Schema instead of being closed with `additionalProperties: false`.
