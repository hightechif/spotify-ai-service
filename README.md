# Spotify AI Assistant Gateway (`spotify-ai-service`)

This directory houses the backend gateway service for the Spotify AI Assistant. It is implemented in Python 3.11+ using **FastAPI**, **LangChain**, and the **MCP Python SDK**.

The gateway acts as the orchestrator: it establishes a connection to the local `spotify-mcp-server` over a stdio subprocess, routes user requests from the Android app to a local LLM running on Ollama, and manages real-time status and playback streaming over WebSockets.

---

## 🚀 Key Features

* **WebSocket Agent Streaming**: A reactive `/chat` endpoint that streams token-by-token LLM textual responses and tool calling lifecycle states (e.g. `Thinking...`, `Calling tool search_spotify...`).
* **Dynamic Path Resolution**: Automatically locates the adjacent `spotify-mcp-server` and virtual environment without hardcoding absolute paths. Can be overridden in `.env` if needed.
* **Asynchronous Playback Sync**: Runs a background async task that polls the Spotify API every 2 seconds and broadcasts playback events (track, artist, album art, play status, volume, device) down to the client.
* **Strict Type Safety**: Fully type-annotated code validated by `mypy`.
* **Clean Session State**: Manages conversational history memory independently per active WebSocket session.

---

## 🛠️ Prerequisites

1. **Python 3.11+**
2. **uv** (Recommended Python package manager):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
3. **Ollama**: Running locally with the `qwen2.5:3b` model:
   ```bash
   ollama run qwen2.5:3b
   ```

---

## ⚙️ Environment Configuration

Copy the example configuration file and fill in your developer credentials:

```bash
cp .env.example .env
```

### Configuration Parameters

| Key | Description | Default / Example |
| :--- | :--- | :--- |
| `SPOTIFY_CLIENT_ID` | Your Spotify developer client ID | `your_spotify_client_id_here` |
| `SPOTIFY_CLIENT_SECRET` | Your Spotify developer client secret | `your_spotify_client_secret_here` |
| `SPOTIFY_REFRESH_TOKEN` | Generated Spotify refresh token | `your_spotify_refresh_token_here` |
| `OLLAMA_MODEL` | The Ollama model name to use | `qwen2.5:3b` |
| `SPOTIFY_MCP_SERVER_DIR` | (Optional) Path override for the MCP server folder | *Adjacent directory* |
| `SPOTIFY_MCP_PYTHON_PATH` | (Optional) Path override for the virtualenv python interpreter | *Adjacent directory virtualenv* |
| `SPOTIFY_REDIRECT_PORT` | Port to use for Spotify callback authorization | `8888` |

---

## 🏃 Running the Server

### 1. Install Dependencies
Sync project dependencies using `uv`:
```bash
uv sync
```

### 2. Start the Service
Start the FastAPI server using `uvicorn`. Make sure to bind to `0.0.0.0` so that external devices (like your physical Android phone on the same WiFi network) can access the service:

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## 🧪 Testing & Verification

### Static Type Checks
Run type diagnostics using `mypy`:
```bash
uv run mypy main.py mcp_client.py
```

### WebSocket Integration Test
We provide a standalone python script to test WebSocket connections, status notifications, and agent tool execution. Run it with:
```bash
uv run python path/to/scratch/test_websocket.py
```
