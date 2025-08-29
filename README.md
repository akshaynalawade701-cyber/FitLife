# FitLife — Fitness Website

A fast, responsive static site with calculators, a workout plan builder, and a nutrition tracker. All data stays local in your browser.

## Features
- BMI, BMR, TDEE calculators (metric/imperial)
- Workout builder (goal, experience, days/week, equipment) with save/print
- Nutrition tracker with goals, daily meals, progress bars, import/export (JSON)
- Body Scan (beta): posture/imbalance metrics with on-photo overlays
- Light/dark theme toggle; mobile-friendly navigation
- Privacy and terms pages

## Run locally
From the project root:

```bash
cd /workspace/fitness-website
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

## Offline & PWA
- This site includes a Service Worker for offline caching and a Web App Manifest.
- On first load, assets are cached. Subsequent loads work offline (calculators, workout builder, nutrition tracker, and static pages). The Body Scan needs internet to fetch ML models from the CDN if not already cached.
- To test:
  1. Load `http://localhost:8000` once while online.
  2. Open DevTools → Application → Service Workers and confirm `sw.js` is active.
  3. Toggle “Offline” in the Network tab and reload the page; the app should still work.

Notes:
- Some browsers require HTTPS for Service Workers; `localhost` is allowed for development.
- If you change files, the SW will update automatically on refresh; you can also call `navigator.serviceWorker.getRegistration().then(r=>r&&r.update())` in the console.

## Deploy to GitHub Pages
1. Create a new public repo on GitHub (e.g., `fitness-website`).
2. Push the contents of this folder to the repo root (so `index.html` is at the top level).
3. Ensure a file named `.nojekyll` exists at the repo root (included here).
4. In GitHub → repo Settings → Pages:
   - Source: Deploy from a branch
   - Branch: `main` (or `master`), folder `/ (root)`
5. Save. Your site will be available at `https://<your-username>.github.io/<repo-name>/`.

Optionally add a custom domain under Pages settings.

## Tech
- HTML, CSS, JavaScript (no frameworks)
- LocalStorage for persistence; no backend required

## License
MIT