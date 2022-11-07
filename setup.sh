mkdir -p ~/.streamlit/

echo "[theme]
base='light'
backgroundColor='#f1e7e7'
secondaryBackgroundColor='#D7D8D7'
[server]
headless = true
port = $PORT
enableCORS = false"  > ~/.streamlit/config.toml