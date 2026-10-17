# NSE F&O Hourly CPR + 9:15 AM ATM±3 O=H/O=L Web & Mobile Package

This package contains everything you need to access your **Hourly CPR + 9:15 AM Opening Drive + ATM±3 O=H/O=L Scanner** from anywhere online or on your mobile phone (Android / iOS) for **100% FREE**.

## Package Contents
1. `index.html` & `manifest.json` — Standalone, zero-server **Interactive Web & Mobile PWA Dashboard** (works offline/online on any phone or browser, or via free static hosting like **Netlify Drop** or **GitHub Pages**).
2. `streamlit_app.py` — Full live Python web server dashboard powered by **Streamlit** + `yfinance` for 1-click free cloud hosting on **Streamlit Community Cloud**.
3. `nse_fno_options_scanner.py` — Upgraded CLI & engine script with **Hourly CPR** (`compute_hourly_cpr_from_intraday`), **F&O Ban & Low-Liquidity Mid-Cap Skipper**, and `--watch 60` live terminal loop.
4. `requirements.txt` — Python dependencies automatically installed by Streamlit Cloud.

---

## Option A: Fastest Zero-Code Mobile/Web Hosting (30 Seconds via Netlify Drop — 100% Free)
If you want an instant URL (`https://your-scanner.netlify.app`) that opens on your phone like a native mobile app:
1. Unzip this package on your computer or phone.
2. Go to **[app.netlify.com/drop](https://app.netlify.com/drop)** in your browser.
3. Drag and drop the unzipped folder onto the page.
4. Netlify immediately gives you a permanent HTTPS link! Open that link on your Android (Chrome) or iPhone (Safari) and tap **"Add to Home Screen"** to install it as a full-screen mobile app.

---

## Option B: Free Live-Data Python Cloud Hosting (Streamlit Community Cloud — 100% Free)
To run the live `yfinance` Python backend in the cloud and access it from any mobile browser:
1. Create a free account at **[github.com](https://github.com)** and create a new repository (e.g., `nse-hourly-cpr-scanner`).
2. Upload the 4 files (`streamlit_app.py`, `nse_fno_options_scanner.py`, `requirements.txt`, `index.html`) to your GitHub repository.
3. Go to **[share.streamlit.io](https://share.streamlit.io)**, sign in with GitHub, and click **"Create app"**.
4. Select your repository and set **Main file path** to `streamlit_app.py`, then click **Deploy!**.
5. In ~60 seconds, you will get a free `https://your-app.streamlit.app` URL that fetches live NSE 15m/1H data on demand from your phone or laptop!

---

## Option C: Run Directly on Android / PC Terminal
```bash
pip install -r requirements.txt
# Run Live 9:15 AM + Intraday Hourly CPR Scan (auto-refresh every 60s, skipping F&O Ban & mid-caps):
python nse_fno_options_scanner.py --mode all --interval 15m --ban-list "ZEEL,PNB,SAIL,BIOCON" --watch 60
```
