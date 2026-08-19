# AI Diabetic Retinopathy Screening System

Unified SIH integration project containing the React frontend, the existing
chatbot intake module, and the existing five-class Keras retinal model.
The backend is one FastAPI application: it exposes `/chat`, `/session/{id}`,
`/screening/predict`, and `/screening/explain`.

---

## 1. Folder Structure

```
SIH-26/
├── backend/
│   ├── main.py                    # Unified FastAPI app
│   ├── routes/screening.py        # Prediction and Grad-CAM endpoints
│   ├── services/model_service.py
│   └── services/nlp_service.py
├── backend/core/                  # Session manager and validation
├── backend/models/                # Chatbot schemas
├── ml/
│   ├── dr_model.py                # Existing model wrapper + Grad-CAM
│   └── models/prototype_model.keras
├── frontend/
│   ├── package.json
│   └── src/
│       ├── api/                   # Backend adapters
│       ├── pages/                 # Screening, analysis, assistant, reports
│       └── components/
└── README.md
```

---

## 2. Installation Steps

Prerequisites: **Python 3.11–3.13**, **Node.js 18+**, and the TensorFlow runtime
supported by the installed Python version. The current chatbot backend uses
the updated rule-based NLP service, so no frontend secret is required.

### Backend setup

```powershell
cd C:\Users\devasri\OneDrive\Desktop\projects\SIH-26
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
Copy-Item backend\.env.example backend\.env
```

For real EfficientNet prediction and Grad-CAM, use Python 3.11–3.13 and
install the optional ML runtime as well:

```powershell
pip install -r backend/requirements-ml.txt
```

Edit `backend/.env` as needed:

```
MODEL_PATH=ml/models/prototype_model.keras
DATA_DIR=data
ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

Run the backend:

```powershell
uvicorn backend.main:app --reload --port 8000
```

Run this command from the repository root, not from `frontend` or `backend`.
The chatbot endpoints work without TensorFlow; image prediction and Grad-CAM
require a Python version supported by the installed TensorFlow package.

The API is now live at `http://localhost:8000`. Interactive docs at
`http://localhost:8000/docs`.

### Frontend setup

```bash
cd frontend
npm install
cp .env.example .env
```

Edit `frontend/.env` if your backend runs somewhere other than
`http://localhost:8000`:

```
VITE_USE_MOCK_API=false
VITE_API_BASE_URL=http://localhost:8000
```

Run the frontend:

```bash
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## 3. Environment Variables

| File               | Variable            | Description                                   |
|--------------------|----------------------|------------------------------------------------|
| backend/.env       | `MODEL_PATH`         | Existing Keras model path                       |
| backend/.env       | `DATA_DIR`           | Folder for session JSON files (default `data`)  |
| backend/.env       | `ALLOWED_ORIGINS`    | Comma-separated CORS origins                    |
| frontend/.env      | `VITE_API_BASE_URL`  | URL of the FastAPI backend                      |

---

## 4. Run Commands (quick reference)

```bash
# Terminal 1 — backend (from repository root)
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — frontend
cd frontend && npm run dev
```

---

## 5. API Reference

### `POST /screening/predict`

Multipart form field: `image`. Returns the actual five probabilities,
predicted grade, severity label, and model confidence.

### `POST /screening/explain`

Multipart form field: `image`. Returns the prediction plus base64 data URIs
for the original image, Grad-CAM heatmap, and overlay.

### `POST /chat`

Request:
```json
{ "session_id": "abc123", "message": "I am 56 years old and diabetic for 12 years." }
```

Response:
```json
{
  "reply": "Thanks. Do you know your most recent HbA1c value?",
  "patient_state": {
    "session_id": "abc123",
    "structured_data": { "age": 56, "diabetes_duration": 12, "...": null },
    "dynamic_data": {},
    "conversation_history": [ { "role": "user", "message": "...", "timestamp": "..." } ],
    "completed": false
  },
  "completed": false
}
```

### `GET /session/{session_id}`

Returns the full stored session JSON (same shape as `patient_state` above).

---

## 6. Testing Instructions

### Manual test via curl

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test1","message":"I am 56 years old and diabetic for 12 years."}'

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test1","message":"My HbA1c was around 9.2 last month, and my blood pressure was 130/85."}'

curl http://localhost:8000/session/test1
```

The second call above should include all four required fields
(`age`, `diabetes_duration`, `hba1c`, `blood_pressure`) and the response
should return `"completed": true`.

### Manual test via UI

1. Start backend and frontend as described above.
2. Open the app; the assistant greets you automatically.
3. Type natural sentences, e.g.:
   - "I am 56 years old and diabetic for 12 years."
   - "My HbA1c was around 9.2 last month."
   - "My vision gets blurry at night, and my father also had diabetic retinopathy."
   - "My blood pressure yesterday was 130/85."
4. Confirm the chat becomes complete once the four required fields are known.
5. Refresh the page — the conversation should reload from the stored session
   (via `localStorage` session id + `GET /session/{id}`).
6. Click **New Session** to start over with a fresh `session_id`.

### Error-handling checks

- Stop the backend and send a message → frontend shows a network error banner.
- Remove `GROK_API_KEY` from `.env` and restart backend → `/chat` returns
  a 502 with a clear "GROK_API_KEY is not set" message.
- Manually corrupt a `backend/data/session_<id>.json` file (e.g. truncate it)
  → next request for that session starts fresh instead of crashing, and the
  corrupted file is backed up alongside it.
- Send `{"session_id": "", "message": "hi"}` → 400 error, "session_id is required."

---

## 7. Data Model Notes

Each session is stored as `backend/data/session_<session_id>.json`:

```json
{
  "session_id": "abc123",
  "structured_data": { "age": 56, "hba1c": 9.2, "...": null },
  "dynamic_data": { "family_history": "Father had diabetic retinopathy" },
  "conversation_history": [ { "role": "user", "message": "...", "timestamp": "..." } ],
  "completed": true,
  "pending_confirmation": null
}
```

This file is the export format consumed by the downstream DR screening
AI pipeline — no database is used, and no information outside the core
fields is ever discarded (it lands in `dynamic_data`).
