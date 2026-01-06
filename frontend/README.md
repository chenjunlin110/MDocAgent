# Frontend

## Prerequisites

- Node.js 18+ (required by Next.js 14)

## Setup

From the repo root:

```bash
cd frontend
npm install
cp .env.local.example .env.local
```

Edit `.env.local` to point at your backend:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Run (dev)

```bash
npm run dev
```

Open http://localhost:3000.

## Build (prod)

```bash
npm run build
npm start
```
