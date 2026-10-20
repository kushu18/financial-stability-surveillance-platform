# RBI Command Portal

This is a standalone front-end shell for the Financial Stability Surveillance Platform.

## What it is

- A professional, finance-grade landing portal for the 33-bank Financial Contagion Monitor concept.
- A separate static site that does not depend on your future project code.
- A place where you can later mount your real dashboard, charts, model outputs, or APIs.

## Files

- `index.html` - the page structure
- `styles.css` - the institutional visual system and RBI watermark treatment
- `script.js` - interactivity for crisis replay, bank explorer, and project dock modes
- `assets/rbi-circle.png` - the RBI emblem used in the background and header

## How to use

1. Open `index.html` directly in a browser for a quick preview.
2. If you want a local server, run:

```powershell
python -m http.server 8080
```

Then open `http://localhost:8080/rbi-command-portal/`

## How to attach your project later

- Replace the placeholder panel inside the `Project Dock` section with your real app.
- You can embed your project as:
  - an iframe
  - a mounted frontend root
  - a routed sidecar experience
- The rest of the shell can stay unchanged as the supervisory and presentation layer.
