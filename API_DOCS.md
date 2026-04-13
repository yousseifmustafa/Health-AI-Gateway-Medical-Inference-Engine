# SehaBot Medical AI — API Documentation

> **Version:** 2.0 &nbsp;|&nbsp; **Protocol:** REST + SSE &nbsp;|&nbsp; **Auth:** API Key *(roadmap)*

SehaBot is a multimodal Medical AI suite built on FastAPI + LangGraph. It exposes three
production-ready HTTP endpoints covering real-time conversational triage, prescription OCR,
and medicine packaging analysis.

---

## Table of Contents

1. [Base URL & Environments](#1-base-url--environments)
2. [Endpoint: Stateful Multimodal Chat — `POST /chat`](#2-endpoint-stateful-multimodal-chat----post-chat)
3. [Endpoint: Prescription Analyzer — `POST /analyze/prescription`](#3-endpoint-prescription-analyzer----post-analyzeprescription)
4. [Endpoint: Medicine Box Analyzer — `POST /analyze/medicine-box`](#4-endpoint-medicine-box-analyzer----post-analyzemedicine-box)
5. [Health Check — `GET /health`](#5-health-check----get-health)
6. [Global Error Handling](#6-global-error-handling)
7. [Integration Examples](#7-integration-examples)
8. [Deployment Notes](#8-deployment-notes)

---

## 1. Base URL & Environments

| Environment | Base URL |
|-------------|----------|
| **Local Development** | `http://localhost:8000` |
| **Staging** | `https://staging-api.sehabot.com` |
| **Production** | `https://api.sehabot.com` |

All endpoints described in this document are relative to the base URL.

**Switching environments:** Define `BASE_URL` as a single constant in your client
code and update it per deployment target — see [§8 Deployment Notes](#8-deployment-notes).

---

## 2. Endpoint: Stateful Multimodal Chat — `POST /chat`

### Description

Real-time AI medical triage assistant with persistent conversation memory (per `user_id`).
Supports plain-text queries and optionally a medical image (X-ray, prescription, drug box).

The response is a **Server-Sent Events (SSE)** stream — the server pushes JSON lines
progressively as the agent reasons, so the frontend can render tokens in real time.

> **Important:** This endpoint is **stateful**. The agent remembers the full conversation
> history across multiple calls for the same `user_id`. Each call appends to the session.
> To start a fresh session, simply use a new `user_id`.

---

### Request

| Property | Value |
|---|---|
| **Method** | `POST` |
| **Path** | `/chat` |
| **Content-Type** | `multipart/form-data` |
| **Rate Limit** | 10 requests / minute per IP |

#### Form Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `query` | `string` | Yes | The patient's question or symptom description. Max 5,000 characters. |
| `user_id` | `string` | Yes | Unique patient identifier. Alphanumeric, hyphens, underscores. 1-255 chars. Used to route to the correct memory thread. |
| `summary` | `string` | No | Previous conversation summary to seed the session context. Defaults to `"No Summary Found."` |
| `thread_id` | `string` | No | Legacy field, currently ignored — thread routing is handled server-side via `user_id`. |
| `image` | `file` | No | A medical image (`image/jpeg` or `image/png`). When provided, the agent activates its vision analysis tool automatically. |

---

### Response — SSE Stream

The response is a `text/event-stream` where each line is a **newline-delimited JSON object**.
The client must read lines incrementally and parse each one independently.

#### Event Types

| `type` | When | Payload Fields |
|---|---|---|
| `"status"` | Agent enters a processing node | `node` (string), `content` (Arabic status label) |
| `"token"` | LLM generates the next word/chunk | `content` (string) — append to display buffer |
| `"final"` | Stream complete | `final_answer` (full string), `summary` (updated session summary) |
| `"error"` | Unhandled agent exception | `content` (error description string) |
| `"action_required"` | Agent paused for human approval (e.g. family notification) | `content`, `tool` (tool name that triggered the pause) |

#### Example Stream Sequence

```
{"type": "status", "node": "conflict_detector_node", "content": "جاري التحقق من تعارض المعلومات..."}
{"type": "status", "node": "agent_node", "content": "جاري صياغة الرد الطبي..."}
{"type": "token",  "content": "عندك"}
{"type": "token",  "content": " ألم"}
{"type": "token",  "content": " في الصدر"}
{"type": "final",  "final_answer": "عندك ألم في الصدر...", "summary": "Patient reported chest pain..."}
```

---

### cURL Example

```bash
# Text-only query
curl -X POST "http://localhost:8000/chat" \
  -F "query=عندي صداع شديد من أمس" \
  -F "user_id=patient_001"

# With medical image
curl -X POST "http://localhost:8000/chat" \
  -F "query=ممكن تحلل الصورة دي؟" \
  -F "user_id=patient_001" \
  -F "image=@/path/to/xray.jpg"
```

---

## 3. Endpoint: Prescription Analyzer — `POST /analyze/prescription`

### Description

**Stateless**, one-shot OCR endpoint. Accepts a single prescription image (handwritten or
printed) and returns a structured JSON object containing all extracted medication data.

> This endpoint does **not** use LangGraph, session memory, or a thread ID. Each call
> is fully independent.

---

### Request

| Property | Value |
|---|---|
| **Method** | `POST` |
| **Path** | `/analyze/prescription` |
| **Content-Type** | `multipart/form-data` |
| **Rate Limit** | 20 requests / minute per IP |

#### Form Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `image` | `file` | Yes | A clear photo of the prescription. Accepted MIME types: `image/jpeg`, `image/png`. |

---

### Response Schema — `PrescriptionResponse`

| Field | Type | Nullable | Description |
|---|---|---|---|
| `patient_name` | `string` | Yes | Patient name as written on the prescription. |
| `patient_date` | `string` | Yes | Date printed or written on the prescription. |
| `doctor_name` | `string` | Yes | Prescribing physician's name. |
| `medications` | `MedicationEntry[]` | — | Array of extracted medication entries (see below). |
| `general_notes` | `string` | Yes | Free-text doctor instructions not tied to a specific drug. |
| `confidence` | `"HIGH" or "MEDIUM" or "LOW"` | — | OCR quality assessment. `HIGH` = fully legible; `LOW` = blurry/unclear. |
| `unreadable_parts` | `string` | Yes | Description of sections that could not be parsed. |

#### `MedicationEntry` Object

| Field | Type | Nullable | Description |
|---|---|---|---|
| `name` | `string` | — | Medication name exactly as written (trade or generic). |
| `dosage` | `string` | — | Amount per administration (e.g. `"500mg"`, `"1 tablet"`). |
| `frequency` | `string` | — | How often to take (e.g. `"twice daily"`, `"مرتين يومياً"`). |
| `duration` | `string` | Yes | Course length if specified (e.g. `"7 days"`). |
| `notes` | `string` | Yes | Special instructions for this drug (e.g. `"take with food"`). |

---

### Example Response

```json
{
  "patient_name": "Mohammed Al-Rashidi",
  "patient_date": "2024-03-15",
  "doctor_name": "Dr. Sarah Hassan",
  "medications": [
    {
      "name": "Amoxicillin",
      "dosage": "500mg",
      "frequency": "Three times daily",
      "duration": "7 days",
      "notes": "Take with food"
    },
    {
      "name": "Ibuprofen",
      "dosage": "400mg",
      "frequency": "Twice daily",
      "duration": "5 days",
      "notes": "Take after meals"
    },
    {
      "name": "Omeprazole",
      "dosage": "20mg",
      "frequency": "Once daily",
      "duration": null,
      "notes": "Take 30 minutes before breakfast"
    }
  ],
  "general_notes": "Rest and increase fluid intake. Follow up in 1 week if symptoms persist.",
  "confidence": "HIGH",
  "unreadable_parts": null
}
```

### cURL Example

```bash
curl -X POST "http://localhost:8000/analyze/prescription" \
  -F "image=@/path/to/prescription.jpg"
```

---

## 4. Endpoint: Medicine Box Analyzer — `POST /analyze/medicine-box`

### Description

**Stateless**, one-shot packaging intelligence endpoint. Accepts a photo of a medicine box
or blister pack and returns a structured JSON object with pharmaceutical data extracted
from the packaging.

> This endpoint does **not** use LangGraph, session memory, or a thread ID. Each call
> is fully independent.

---

### Request

| Property | Value |
|---|---|
| **Method** | `POST` |
| **Path** | `/analyze/medicine-box` |
| **Content-Type** | `multipart/form-data` |
| **Rate Limit** | 20 requests / minute per IP |

#### Form Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `image` | `file` | Yes | A clear photo of the medicine packaging. Accepted MIME types: `image/jpeg`, `image/png`. |

---

### Response Schema — `MedicineBoxResponse`

| Field | Type | Nullable | Description |
|---|---|---|---|
| `trade_name` | `string` | — | Brand/commercial name as printed on the packaging. |
| `generic_name` | `string` | Yes | INN (International Non-proprietary Name) / active substance name. |
| `active_ingredients` | `string[]` | — | List of active chemical compounds with concentrations. |
| `concentration` | `string` | Yes | Formulation strength (e.g. `"500mg/5ml"`, `"10mg"`). |
| `dosage_form` | `string` | Yes | Physical form: `"Tablet"`, `"Syrup"`, `"Injection"`, etc. |
| `indications` | `string[]` | — | Stated therapeutic uses / indications from the packaging. |
| `contraindications` | `string[]` | — | Stated warnings, contraindications, or cautions. |
| `manufacturer` | `string` | Yes | Manufacturing company name. |
| `storage_conditions` | `string` | Yes | Storage instructions as printed (e.g. `"Store below 25°C"`). |
| `expiry_date` | `string` | Yes | Expiry date if legible on the image. |

---

### Example Response

```json
{
  "trade_name": "Augmentin",
  "generic_name": "Amoxicillin/Clavulanate",
  "active_ingredients": [
    "Amoxicillin 875mg",
    "Clavulanic acid 125mg"
  ],
  "concentration": "875mg/125mg",
  "dosage_form": "Film-coated Tablet",
  "indications": [
    "Respiratory tract infections",
    "Urinary tract infections",
    "Skin and soft tissue infections",
    "Dental infections"
  ],
  "contraindications": [
    "Hypersensitivity to penicillins or beta-lactams",
    "History of Augmentin-associated jaundice",
    "Severe renal impairment (adjust dose)"
  ],
  "manufacturer": "GlaxoSmithKline",
  "storage_conditions": "Store below 25°C. Keep out of reach of children.",
  "expiry_date": "06/2026"
}
```

### cURL Example

```bash
curl -X POST "http://localhost:8000/analyze/medicine-box" \
  -F "image=@/path/to/medicine_box.jpg"
```

---

## 5. Health Check — `GET /health`

A lightweight probe for load balancers and uptime monitors.

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "graph_loaded": true
}
```

`graph_loaded: false` means the LangGraph supervisor is still compiling (normal for 5-10s
after cold start). The `/chat` endpoint will return `503` until this becomes `true`.

---

## 6. Global Error Handling

All endpoints return standard HTTP status codes. Non-`2xx` responses always include a
JSON body with a `detail` field.

| HTTP Code | Meaning | When It Occurs | Recommended Client Action |
|---|---|---|---|
| `200 OK` | Success | Request processed without error. | Parse body normally. |
| `400 Bad Request` | Invalid image content | Image uploaded to an analyzer endpoint does not match the expected category (e.g., a selfie uploaded as a prescription). | Show user-facing message asking them to upload the correct image type. |
| `422 Unprocessable Entity` | Validation error | Missing required field, `user_id` format invalid, query too long (>5,000 chars), or empty image file uploaded. | Display the `detail` field to guide the user to correct input. |
| `429 Too Many Requests` | Rate limit exceeded | More than 10 req/min (chat) or 20 req/min (analyzers) from the same IP. | Back off and retry after 60 seconds. Show a countdown to the user. |
| `500 Internal Server Error` | Unhandled server exception | Unexpected crash in application logic. | Log the error, display a generic retry message to the user. |
| `502 Bad Gateway` | AI analysis failure | The LLM call to Gemini failed or timed out inside an analyzer endpoint. | Retry once; if it persists, report to the backend team. |
| `503 Service Unavailable` | Graph not compiled | Cold start — LangGraph supervisor is still initialising. | Poll `/health` until `graph_loaded: true`, then retry. |

#### Error Response Body

```json
{
  "detail": "Query exceeds maximum length of 5000 characters."
}
```

---

## 7. Integration Examples

### Python (`requests`)

```python
import requests
import json
import os

BASE_URL = os.getenv("SEHABOT_API_URL", "http://localhost:8000")


# ── 1. Stateful Chat (Streaming) ─────────────────────────────────────────────

def chat_stream(user_id: str, query: str, image_path: str = None):
    payload = {"query": query, "user_id": user_id}
    files   = {}
    if image_path:
        files["image"] = open(image_path, "rb")

    with requests.post(
        f"{BASE_URL}/chat",
        data=payload,
        files=files or None,
        stream=True,
        timeout=180,
    ) as resp:
        resp.raise_for_status()
        full_response = ""
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            event = json.loads(raw_line.decode("utf-8"))
            if event["type"] == "token":
                print(event["content"], end="", flush=True)
                full_response += event["content"]
            elif event["type"] == "final":
                print()  # newline
                return full_response, event.get("summary", "")
            elif event["type"] == "error":
                raise RuntimeError(f"Agent error: {event['content']}")


# ── 2. Prescription Analyzer (One-Shot) ──────────────────────────────────────

def analyze_prescription(image_path: str) -> dict:
    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/analyze/prescription",
            files={"image": f},
            timeout=120,
        )
    resp.raise_for_status()
    return resp.json()


# ── 3. Medicine Box Analyzer (One-Shot) ──────────────────────────────────────

def analyze_medicine_box(image_path: str) -> dict:
    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/analyze/medicine-box",
            files={"image": f},
            timeout=120,
        )
    resp.raise_for_status()
    return resp.json()


# ── Usage ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    answer, summary = chat_stream("patient_001", "I have a severe headache since yesterday")
    print(f"\nSummary: {summary}")

    rx_data = analyze_prescription("./prescription.jpg")
    for med in rx_data["medications"]:
        print(f"  {med['name']} — {med['dosage']} — {med['frequency']}")

    box_data = analyze_medicine_box("./augmentin_box.jpg")
    print(f"Trade name: {box_data['trade_name']}")
    print(f"Ingredients: {', '.join(box_data['active_ingredients'])}")
```

---

### JavaScript / TypeScript (`fetch` — React / Next.js)

```typescript
const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ── Types ─────────────────────────────────────────────────────────────────────

interface MedicationEntry {
  name: string;
  dosage: string;
  frequency: string;
  duration: string | null;
  notes: string | null;
}

interface PrescriptionResponse {
  patient_name:     string | null;
  patient_date:     string | null;
  doctor_name:      string | null;
  medications:      MedicationEntry[];
  general_notes:    string | null;
  confidence:       "HIGH" | "MEDIUM" | "LOW";
  unreadable_parts: string | null;
}

interface MedicineBoxResponse {
  trade_name:         string;
  generic_name:       string | null;
  active_ingredients: string[];
  concentration:      string | null;
  dosage_form:        string | null;
  indications:        string[];
  contraindications:  string[];
  manufacturer:       string | null;
  storage_conditions: string | null;
  expiry_date:        string | null;
}

// ── 1. Stateful Chat with SSE Streaming ──────────────────────────────────────

async function chatStream(
  userId:     string,
  query:      string,
  imageFile:  File | null,
  onToken:    (token: string) => void,
  onComplete: (finalAnswer: string, summary: string) => void,
): Promise<void> {
  const form = new FormData();
  form.append("query",   query);
  form.append("user_id", userId);
  if (imageFile) form.append("image", imageFile);

  const resp = await fetch(`${BASE_URL}/chat`, { method: "POST", body: form });

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(`API ${resp.status}: ${err.detail}`);
  }

  const reader  = resp.body!.getReader();
  const decoder = new TextDecoder();
  let fullAnswer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const lines = decoder.decode(value, { stream: true }).split("\n");
    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const event = JSON.parse(line);
        if (event.type === "token") {
          fullAnswer += event.content;
          onToken(event.content);
        } else if (event.type === "final") {
          onComplete(event.final_answer ?? fullAnswer, event.summary ?? "");
        } else if (event.type === "error") {
          throw new Error(`Agent error: ${event.content}`);
        }
      } catch {
        /* skip malformed lines */
      }
    }
  }
}

// ── 2. Prescription Analyzer ──────────────────────────────────────────────────

async function analyzePrescription(imageFile: File): Promise<PrescriptionResponse> {
  const form = new FormData();
  form.append("image", imageFile);

  const resp = await fetch(`${BASE_URL}/analyze/prescription`, {
    method: "POST",
    body: form,
  });

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(`API ${resp.status}: ${err.detail}`);
  }
  return resp.json() as Promise<PrescriptionResponse>;
}

// ── 3. Medicine Box Analyzer ──────────────────────────────────────────────────

async function analyzeMedicineBox(imageFile: File): Promise<MedicineBoxResponse> {
  const form = new FormData();
  form.append("image", imageFile);

  const resp = await fetch(`${BASE_URL}/analyze/medicine-box`, {
    method: "POST",
    body: form,
  });

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(`API ${resp.status}: ${err.detail}`);
  }
  return resp.json() as Promise<MedicineBoxResponse>;
}
```

---

## 8. Deployment Notes

### Switching `BASE_URL` by Environment

Never hard-code `localhost:8000` in production. Define `BASE_URL` in exactly one place
per client application and let the environment control its value.

#### Python Client

```python
import os
BASE_URL = os.getenv("SEHABOT_API_URL", "http://localhost:8000")
```

```bash
# Production
SEHABOT_API_URL=https://api.sehabot.com python your_script.py

# Staging
SEHABOT_API_URL=https://staging-api.sehabot.com python your_script.py
```

#### React / Next.js

`.env.local` (development):
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

`.env.production`:
```
NEXT_PUBLIC_API_URL=https://api.sehabot.com
```

#### Flutter

```dart
const String kBaseUrl = String.fromEnvironment(
  'API_BASE_URL',
  defaultValue: 'http://localhost:8000',
);
```

```bash
flutter build apk --dart-define=API_BASE_URL=https://api.sehabot.com
```

---

### CORS

If your frontend is hosted on a different origin, add `CORSMiddleware` to `server.py`:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://app.sehabot.com"],  # or ["*"] for development only
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)
```

---

### Server Startup

```bash
# Development (auto-reload)
uvicorn server:app --host 0.0.0.0 --port 8000 --reload

# Production (multiple workers)
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4

# Docker
docker compose up --build
```

---

### Cold Start Behaviour

On server start, the LangGraph supervisor graph compiles and connects to PostgreSQL.
This takes approximately **5-10 seconds**. During this window:

- `GET /health` returns `{"graph_loaded": false}`
- `POST /chat` returns `503 Service Unavailable`
- `POST /analyze/*` endpoints are **immediately available** (stateless, no graph dependency)

Configure your load balancer to use `/health` as a readiness probe and only route
`/chat` traffic after `graph_loaded` is `true`.

---

### Rate Limits Summary

| Endpoint | Limit | Scope |
|---|---|---|
| `POST /chat` | 10 requests / minute | Per client IP |
| `POST /analyze/prescription` | 20 requests / minute | Per client IP |
| `POST /analyze/medicine-box` | 20 requests / minute | Per client IP |

Clients that exceed the limit receive `HTTP 429`. Implement exponential back-off with a
minimum retry delay of **60 seconds**.

---

*Documentation generated for SehaBot Medical AI Suite — SehaTech 2025.*
