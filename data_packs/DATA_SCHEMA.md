# IINTS-AF Data Schema Specification

## Universal Data Contract

All datasets in IINTS-AF must conform to this schema for interoperability.

### Core Schema v1.0

```json
{
  "timestamp": "float",           # Minutes from start
  "glucose": "mg/dL",            # Blood glucose reading
  "carbs": "grams | null",       # Carbohydrate intake (optional)
  "insulin": "units | null",     # Insulin delivery (optional)
  "source": "sensor | simulated" # Data origin
}
```

### Extended Schema v1.1 (Optional Fields)

```json
{
  "timestamp": "float",
  "glucose": "mg/dL", 
  "carbs": "grams | null",
  "insulin": "units | null",
  "source": "sensor | simulated",
  "confidence": "float | null",   # Sensor confidence (0-1)
  "noise_level": "float | null",  # Added noise magnitude
  "patient_id": "string | null",  # Anonymous patient identifier
  "scenario": "string | null"     # Scenario type
}
```

### Metadata Requirements

Each data pack must include `metadata.yaml`:

```yaml
name: "Dataset Name"
version: "1.0"
schema_version: "1.1"
description: "Brief description"
source: "synthetic | public | anonymized"
patient_count: 1
duration_hours: 8
scenarios: ["standard_meal", "hyperglycemia"]
license: "Educational use only"
created: "2024-01-08"
```

### Data Pack Structure

```
data_pack_name/
 metadata.yaml       # Dataset metadata
 schema.json        # JSON schema validation
 data.csv          # Main dataset
 README.md         # Documentation
 validation.py     # Optional validation script
```

### Supported Formats

- **CSV**: Standard comma-separated values
- **JSON**: Structured data with nested objects
- **Parquet**: Compressed columnar format (advanced)

### Quality Requirements

1. **No missing timestamps**
2. **Glucose values: 20-600 mg/dL**
3. **Insulin values: 0-50 units**
4. **Consistent time intervals**
5. **No personally identifiable information**

### Import Validation

All datasets are validated on import:
- Schema compliance
- Value range checks
- Temporal consistency
- Privacy screening