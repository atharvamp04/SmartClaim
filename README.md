# 🚗 SmartClaim — AI-Powered Vehicle Insurance Claim Management System

> An end-to-end insurance claim platform with multi-model AI fraud detection, YOLO-based damage analysis, a role-based workflow (Customer → Surveyor → Admin), real-time chat, and automated email notifications.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [ML Models](#ml-models)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
  - [Environment Variables](#environment-variables)
- [Role-Based Workflow](#role-based-workflow)
- [API Reference](#api-reference)
- [Screenshots](#screenshots)
- [Known Issues & Notes](#known-issues--notes)

---

## Overview

SmartClaim is a full-stack vehicle insurance claim management system that uses **three AI models in a fusion pipeline** to automatically detect fraud, estimate repair costs, and assess damage from uploaded car images. It supports three user roles — **Customer**, **Surveyor**, and **Admin** — each with a dedicated dashboard and workflow.

---

## Features

### 🤖 AI / ML Pipeline
- **YOLOv8 Car Parts Detection** — identifies damaged car components (bumper, hood, door, etc.)
- **YOLOv8 Car Damage Detection** — classifies damage types (scratch, dent, crack, broken)
- **Mask R-CNN CNN Model** — calculates per-image damage percentage with annotated visualizations
- **XGBoost Tabular Fraud Model** — scores fraud probability using policyholder metadata (history, age, vehicle type, etc.)
- **Weighted Fusion Engine** — combines all three model scores into a final fraud verdict + risk level
- **Auto Claim Amount Calculation** — looks up part prices from a PostgreSQL pricing catalog; falls back to hardcoded estimates
- **Deduplication** — filters duplicate part detections across multiple images before pricing

### 👤 Customer Portal
- Submit claim with up to 10 vehicle damage photos
- See AI analysis results (fraud score, risk level, damage %, claim amount, annotated images)
- Track claim status through the full lifecycle
- Receive in-platform notifications and email updates
- Re-submit appeal for rejected claims

### 🔍 Surveyor Dashboard
- View assigned claims with full AI analysis context
- Submit field inspection report (notes, recommendation, assessed amount, damage verified flag)
- **Upload field photos** from the inspection site
- Real-time chat with the customer
- Schedule / confirm appointments

### 🛡️ Admin Dashboard
- Overview of all claims with filtering by status and risk level
- Assign surveyors to individual claims
- Make final Approve / Reject decision with rejection reason
- Auto-generated **PDF report** for each claim decision
- Automated email to customer on every status change

### 📧 Notifications
- Gmail SMTP email on claim submission, approval, and rejection
- Customer notification feed inside the portal

---

## Architecture

```
Browser (Next.js 15)
        │
        │  /api/* proxied by Next.js rewrites
        ▼
Django REST Framework (port 8000)
        │
        ├── JWT Authentication (SimpleJWT)
        ├── Claim Submission → ML Pipeline
        │       ├── YOLOv8 Parts Model
        │       ├── YOLOv8 Damage Model
        │       ├── Mask R-CNN CNN Model
        │       └── XGBoost Tabular Model
        │               └── Fusion → Fraud Verdict + Claim Amount
        ├── PostgreSQL Database
        │       ├── Policyholders & Users
        │       ├── Claims, ClaimImages, ClaimHistory
        │       ├── SurveyorFieldPhotos
        │       ├── Chat Messages & Appointments
        │       └── Vehicle Parts Pricing Catalog
        └── Gmail SMTP (email notifications)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 15 (React 19, TypeScript, Turbopack) |
| UI Components | shadcn/ui + Radix UI + Tailwind CSS v4 |
| Backend | Django 5 + Django REST Framework |
| Auth | JWT via `djangorestframework-simplejwt` |
| Database | PostgreSQL (via `psycopg2`) |
| ML — Tabular | scikit-learn + XGBoost + joblib |
| ML — Vision | YOLOv8 via `ultralytics`, Mask R-CNN via PyTorch |
| PDF Generation | ReportLab |
| Email | Django SMTP → Gmail App Password |
| Image Storage | Django Media Files (local disk) |
| Config | `python-decouple` (`.env`) |

---

## ML Models

All models live in `backend/models/`:

| File | Purpose | Size |
|---|---|---|
| `final_best_model.pkl` | XGBoost tabular fraud classifier | ~1.4 MB |
| `scaler_final.pkl` | StandardScaler for tabular features | ~2 KB |
| `label_encoders.pkl` | LabelEncoders for categorical features | ~7 KB |
| `feature_names.pkl` | Feature name list (must match training) | ~1 KB |
| `final_thresholds.pkl` | Optimal decision thresholds per strategy | ~1 KB |
| `maskrcnn_damage_detection.pth` | CNN for pixel-level damage % | ~168 MB |
| `yolov8_car_parts.pt` | YOLOv8 — detects car part regions | ~6 MB |
| `yolov8_car_damage.pt` | YOLOv8 — classifies damage types | ~6 MB |

> **Note:** Models are lazy-loaded on the first prediction request. On a CPU-only machine the first request may take 30–90 seconds while models load into memory.

---

## Project Structure

```
SmartClaim/
├── backend/
│   ├── core/
│   │   ├── settings.py
│   │   ├── urls.py
│   │   ├── middleware.py        # AppendSlash middleware (fixes POST redirect)
│   │   └── wsgi.py
│   ├── detection/
│   │   ├── models.py            # Claim, ClaimImage, SurveyorFieldPhoto, Chat, Appointment
│   │   ├── views.py             # Main ML pipeline + all API views
│   │   ├── claim_views.py       # Claim CRUD + status management
│   │   ├── claim_handler.py     # DB save logic for claims
│   │   ├── report_views.py      # PDF report + customer portal + appeal
│   │   ├── chat_appointment_views.py
│   │   ├── fusion.py            # Weighted fusion of all model scores
│   │   ├── serializers.py
│   │   ├── urls.py
│   │   └── verification_apis.py # DL / RTO / FIR mock verification
│   ├── models/                  # ML model files (.pkl, .pt, .pth)
│   ├── media/                   # Uploaded images (claim + surveyor photos)
│   ├── .env                     # Environment config
│   ├── manage.py
│   └── requirements.txt
│
└── frontend/
    ├── src/
    │   ├── app/
    │   │   ├── page.tsx          # Landing / login gateway
    │   │   ├── login/            # Login page
    │   │   ├── register/         # Registration
    │   │   ├── claim/            # Customer claim submission + AI results
    │   │   ├── customer/         # Customer dashboard + notifications
    │   │   ├── admin/            # Admin dashboard
    │   │   ├── surveyor/         # Surveyor dashboard + report modal
    │   │   └── policyholder/     # Policyholder profile management
    │   └── components/
    │       ├── ui/               # shadcn/ui component library
    │       ├── SurveyorChatAppointment.tsx
    │       └── CustomerChatAppointment.tsx
    ├── next.config.ts            # Proxy rewrites: /api/* → Django :8000
    └── package.json
```

---

## Getting Started

### Prerequisites

- **Python** 3.10+
- **Node.js** 18+ and **npm**
- **PostgreSQL** 14+
- A **Google account** with 2-Step Verification enabled (for email notifications)

---

### Backend Setup

```bash
# 1. Create and activate virtual environment
cd SmartClaim/backend
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate    # macOS / Linux

# 2. Install Python dependencies
pip install -r requirements.txt
pip install torch torchvision    # PyTorch (CPU)
pip install ultralytics          # YOLOv8
pip install scikit-learn xgboost joblib
pip install djangorestframework-simplejwt
pip install django-cors-headers

# 3. Configure environment (see Environment Variables section)
cp .env.example .env   # or create .env manually

# 4. Create the PostgreSQL database
createdb SmartClaim    # or create via pgAdmin

# 5. Run migrations
python manage.py migrate

# 6. Create a superuser (optional)
python manage.py createsuperuser

# 7. Start the dev server
python manage.py runserver 127.0.0.1:8000
```

---

### Frontend Setup

```bash
cd SmartClaim/frontend

# Install dependencies
npm install

# Start the dev server (accessible on local network)
npm run dev -- -H 0.0.0.0

# App runs at http://localhost:3000
```

---

### Environment Variables

Create `backend/.env` with the following values:

```env
# ── Django ──────────────────────────────────────────────
SECRET_KEY=your_django_secret_key_here
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1

# ── Database ─────────────────────────────────────────────
DB_NAME=SmartClaim
DB_USER=postgres
DB_PASSWORD=your_postgres_password
DB_HOST=localhost
DB_PORT=5432

# ── CORS ─────────────────────────────────────────────────
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

# ── Email (Gmail SMTP) ────────────────────────────────────
# Enable 2-Step Verification on your Google account, then create
# an App Password at https://myaccount.google.com/apppasswords
# Paste the 16-character code WITHOUT spaces:
EMAIL_HOST_USER=you@gmail.com
EMAIL_HOST_PASSWORD=yourapppassword16chars

# ── Fraud Detection Thresholds ────────────────────────────
FRAUD_THRESHOLD=0.5
HIGH_RISK_THRESHOLD=0.7
CRITICAL_RISK_THRESHOLD=0.85

# ── Document Verification Boosts ─────────────────────────
NO_POLICE_REPORT_BOOST=0.25
INVALID_FIR_BOOST=0.30
INVALID_VEHICLE_BOOST=0.35
INVALID_LICENSE_BOOST=0.30

# ── Misc ─────────────────────────────────────────────────
USE_MOCK_APIS=True
SMARTCLAIM_PORTAL_URL=http://localhost:3000
```

> **Gmail App Password tip:** Google displays app passwords as `xxxx xxxx xxxx xxxx` for readability, but store them **without spaces** in `.env`.

---

## Role-Based Workflow

```
Customer submits claim
        │
        ▼
AI Pipeline runs (YOLO + CNN + XGBoost + Fusion)
        │
        ▼
Claim saved as "Pending" in database
        │
        ▼
Admin reviews → assigns a Surveyor
        │  (status → "Under Survey")
        ▼
Surveyor visits site → submits field report + photos
        │  (status → "Survey Completed")
        ▼
Admin makes Final Decision: Approve / Reject
        │  (PDF report generated, email sent to customer)
        ▼
Customer sees result in portal
        └── Can appeal a Rejected claim
```

---

## API Reference

All endpoints are prefixed with `/api/detection/`.

### Authentication
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/auth/login/` | Get JWT access + refresh tokens |
| POST | `/api/auth/refresh/` | Refresh access token |
| POST | `/api/detection/register/` | Register new user |
| POST | `/api/detection/login-with-role/` | Login and get role info |

### Claims
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/detection/predict-claim/` | Submit claim + run full AI pipeline |
| GET | `/api/detection/claims/` | List all claims (admin) |
| GET | `/api/detection/claims/<id>/` | Get claim detail |
| GET | `/api/detection/claims/pending/` | Pending claims |
| GET | `/api/detection/claims/high-risk/` | High-risk claims |
| POST | `/api/detection/claims/<id>/status/` | Update status |
| POST | `/api/detection/claims/<id>/decision/` | Admin final decision |
| GET | `/api/detection/claims/<id>/report-pdf/` | Download PDF report |
| POST | `/api/detection/claims/<id>/resubmit/` | Customer appeal |

### Surveyor
| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/detection/surveyor/claims/` | Get assigned claims |
| POST | `/api/detection/surveyor/claims/<id>/report/` | Submit field report + photos |
| POST | `/api/detection/claims/<id>/assign-surveyor/` | Admin assigns surveyor |

### Customer
| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/detection/customer/claims/` | Customer's own claims |
| GET | `/api/detection/customer/notifications/` | Notification feed |

### Chat & Appointments
| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/detection/chat/<claim_id>/messages/` | Get chat history |
| POST | `/api/detection/chat/<claim_id>/send/` | Send a message |
| GET | `/api/detection/appointments/` | List appointments |
| POST | `/api/detection/appointments/create/` | Create appointment |

### System
| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/detection/health/` | Model load status + health check |

---

## Known Issues & Notes

- **First request is slow** — ML models are loaded lazily on the first `/predict-claim` request. On a CPU machine this can take 60–120 seconds. Hit `/api/detection/health/?load=true` after server start to pre-warm the models.
- **Media files** — Uploaded images are stored locally under `backend/media/`. In production, configure an object storage (S3, GCS) instead.
- **Mock document verification** — `USE_MOCK_APIS=True` uses simulated DL / RTO / FIR verification. Set to `False` and provide real API keys (Surepass / Signzy) for production.
- **PostgreSQL required** — The raw SQL pricing queries (`parts_pricing`, `vehicle_makes`, etc.) target PostgreSQL. SQLite is not supported.
- **Gmail App Password** — Must be the 16-character code **without spaces**. Passwords are invalidated if you change your Google account password.

---

## License

This project is built for academic and demonstration purposes.
