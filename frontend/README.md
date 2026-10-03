# ORAIL frontend

A dependency-free, Vercel-deployable frontend for the lab-v1 robotics repository. Place this entire `frontend/` directory at the repository root.

## Run locally

```bash
cd frontend
python3 -m http.server 8080
```

Open http://localhost:8080. Opening index.html directly does not work because the app fetches JSON data.

## Deploy to Vercel

Import your GitHub repository in Vercel. Set **Root Directory** to `frontend`, **Framework Preset** to `Other`, leave **Build Command** empty (override any auto-detected command), and set **Output Directory** to `.`. No install step or environment variables are needed. This folder already contains the static benchmark assets.

## What works

- 24 real development benchmark scenes and original rendered targets.
- Tier filters, selectable scenes, baseline oracle build-order preview, scene JSON download.
- Nine agent roles based on the source configuration.
- Import `record/export.json` and inspect run summaries, hypotheses, experiments, results, and decisions. Imported text is rendered as text, never HTML.
- Responsive mobile layouts, keyboard focus states, descriptive labels, reduced-motion support.

The build-order preview is explicitly a visualization of existing oracle_order data, not a simulated execution. The website does not launch agents or claim live activity. No mock performance metrics are included. No held-out scenes, local credentials, or model call logs are bundled.

## Research records

Run `bash lab.sh report` in the robotics repository and select `record/export.json` with **Import research record**. The import stays in browser memory and is discarded on refresh. Metrics show the latest recorded run, rather than pooling unrelated experiments.

To publish a snapshot, first review the exported record for content you want to share, then copy it to `frontend/data/record.json` and redeploy. The shipped file is an empty array.

## Update benchmark assets

From the repository root:

```bash
python3 frontend/scripts/sync_data.py
```

This only copies development scenes. Research records require explicit export and copying.

## Add live execution later

Keep MuJoCo, CLI authentication, and Omnigent on a separate persistent host. Add an authenticated HTTPS API with scene IDs and validated actions; connect the browser to it. Preserve the lab's method gates, compute budget, locked held-out evaluations, and human approval boundaries. Do not place Claude/Codex credentials in frontend files or Vercel public environment variables.

## Source

Benchmark assets and configured roles: https://github.com/JonathanLehner/Omnigent-Robotics-Lab/tree/lab-v1. This frontend adds an explorer around that lab and is not the Omnigent application.
