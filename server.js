require("dotenv").config();
const express = require("express");
const axios = require("axios");
const WebSocket = require("ws");
const http = require("http");
const path = require("path");
const cors = require("cors");

const FINNHUB_TOKEN = process.env.FINNHUB_API_KEY;
const USE_SANDBOX =
  (FINNHUB_TOKEN && FINNHUB_TOKEN.startsWith("sandbox_")) ||
  process.env.FINNHUB_USE_SANDBOX === "true";
const FINNHUB_API_BASE = USE_SANDBOX
  ? "https://sandbox.finnhub.io/api/v1"
  : "https://finnhub.io/api/v1";
const FINNHUB_WS_URL = FINNHUB_TOKEN
  ? `wss://ws.finnhub.io/?token=${FINNHUB_TOKEN}`
  : null;

const app = express();
app.use(cors());
app.use(express.json());

// Serve static frontend assets
app.use(express.static(path.join(__dirname, "public")));

function generateMockCandles(symbol, from, to, resolutionSec = 1800) {
  const points = [];
  const o = [];
  const h = [];
  const l = [];
  const c = [];
  const t = [];

  const steps = Math.max(5, Math.floor((to - from) / resolutionSec));
  let price = mockPrice || 100;

  for (let i = 0; i < steps; i++) {
    const ts = (from + i * resolutionSec) * 1000;
    const drift = (Math.random() - 0.5) * 0.5;
    const next = Math.max(0.0001, price + drift);
    const high = Math.max(price, next) + Math.random() * 0.2;
    const low = Math.min(price, next) - Math.random() * 0.2;
    const close = next;
    const open = price;

    t.push(Math.floor(ts / 1000));
    o.push(open);
    h.push(high);
    l.push(low);
    c.push(close);

    price = close;
  }

  mockPrice = price;

  return { s: "ok", t, o, h, l, c };
}

// REST route for historical candles
app.get("/api/candles", async (req, res) => {
  const symbol = (req.query.symbol || "AAPL").toUpperCase();
  const resolution = req.query.resolution || "30";

  const nowSec = Math.floor(Date.now() / 1000);
  let from = parseInt(req.query.from, 10);
  let to = parseInt(req.query.to, 10);

  if (Number.isNaN(from) || Number.isNaN(to)) {
    to = nowSec;
    from = nowSec - 5 * 24 * 60 * 60;
  }

  to = Math.min(to, nowSec);
  if (from >= to) {
    from = to - 5 * 24 * 60 * 60;
  }

  if (useMockData) {
    return res.json(generateMockCandles(symbol, from, to, parseInt(resolution, 10) * 60));
  }

  try {
    const response = await axios.get(`${FINNHUB_API_BASE}/stock/candle`, {
      params: {
        symbol,
        resolution,
        from,
        to,
        token: FINNHUB_TOKEN,
      },
    });

    res.json(response.data);
  } catch (err) {
    const status = err.response?.status || 502;
    const data = err.response?.data;
    const finnhubMessage =
      (data && (data.error || data.msg)) || err.message || "Unknown error";
    console.error("Error fetching candles from Finnhub:", data || err.message);

    const hint =
      status === 401 || status === 403
        ? "Cle Finnhub invalide ou abonnement insuffisant. Utilise une cle active ou un token sandbox_ via FINNHUB_API_KEY."
        : undefined;

    if (status === 401 || status === 403) {
      useMockData = true;
      console.warn(
        "Basculage en mode mock pour les candles (cle invalide ou abonnement insuffisant)."
      );
      stopFinnhub();
      startMockTrades();
      return res.json(
        generateMockCandles(symbol, from, to, parseInt(resolution, 10) * 60)
      );
    }

    res.status(status).json({
      error: finnhubMessage,
      hint,
      status,
    });
  }
});

const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

let finnhubSocket = null;
let currentSymbol = "OANDA:EUR_USD";
let useMockData = !FINNHUB_TOKEN;
let mockPrice = 100;
let mockInterval = null;
const lastPriceBySymbol = {};
let allowReconnect = true;

function subscribeToSymbol(symbol) {
  if (useMockData) return;
  if (finnhubSocket && finnhubSocket.readyState === WebSocket.OPEN) {
    console.log("Subscribing to symbol:", symbol);
    finnhubSocket.send(JSON.stringify({ type: "subscribe", symbol }));
  }
}

function connectFinnhub() {
  if (useMockData || !FINNHUB_WS_URL) {
    startMockTrades();
    return;
  }

  console.log("Connecting to Finnhub WebSocket...");
  finnhubSocket = new WebSocket(FINNHUB_WS_URL);

  finnhubSocket.on("open", () => {
    console.log("Connected to Finnhub WebSocket");
    subscribeToSymbol(currentSymbol);
  });

  finnhubSocket.on("message", (data) => {
    // Broadcast raw data from Finnhub to all browser clients
    wss.clients.forEach((client) => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(data.toString());
      }
    });
  });

  finnhubSocket.on("close", () => {
    console.log("Finnhub WebSocket closed. Reconnecting in 3s...");
    if (!useMockData && allowReconnect) {
      setTimeout(connectFinnhub, 3000);
    } else if (useMockData) {
      startMockTrades();
    }
  });

  finnhubSocket.on("error", (err) => {
    console.error("Finnhub WebSocket error:", err.message);
    useMockData = true;
    allowReconnect = false;
    startMockTrades();
    try {
      finnhubSocket.close();
    } catch (_) {
      // ignore
    }
  });
}

connectFinnhub();
if (useMockData) {
  console.warn("No FINNHUB_API_KEY detected. Using mock data for candles and trades.");
  startMockTrades();
}

function stopFinnhub() {
  allowReconnect = false;
  if (finnhubSocket) {
    try {
      finnhubSocket.close();
    } catch (_) {
      // ignore
    }
    finnhubSocket = null;
  }
}

function startMockTrades() {
  if (mockInterval) return;

  mockInterval = setInterval(() => {
    const now = Date.now();
    const symbol = currentSymbol;
    const base = lastPriceBySymbol[symbol] || mockPrice || 100;
    const drift = (Math.random() - 0.5) * 0.4;
    const price = Math.max(0.0001, base + drift);
    lastPriceBySymbol[symbol] = price;
    mockPrice = price;

    const trade = {
      type: "trade",
      data: [
        {
          s: symbol,
          p: Number(price.toFixed(5)),
          v: Math.max(1, Math.floor(Math.random() * 50)),
          t: now,
        },
      ],
    };

    wss.clients.forEach((client) => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(JSON.stringify(trade));
      }
    });
  }, 1000);
}

function stopMockTrades() {
  if (mockInterval) {
    clearInterval(mockInterval);
    mockInterval = null;
  }
}

wss.on("connection", (socket) => {
  console.log("Browser client connected to WebSocket");
  if (useMockData) {
    startMockTrades();
  }

  // Client can request a symbol change: { "type": "set-symbol", "symbol": "TSLA" }
  socket.on("message", (msg) => {
    try {
      const parsed = JSON.parse(msg.toString());
      if (parsed.type === "set-symbol" && typeof parsed.symbol === "string") {
        currentSymbol = parsed.symbol.toUpperCase();
        console.log("Symbol requested by client:", currentSymbol);
        lastPriceBySymbol[currentSymbol] =
          lastPriceBySymbol[currentSymbol] || mockPrice || 100;
        subscribeToSymbol(currentSymbol);
        if (useMockData) {
          startMockTrades();
        }
      }
    } catch (e) {
      console.error("Invalid WS message from client:", e.message);
    }
  });

  socket.on("close", () => {
    console.log("Browser client disconnected from WebSocket");
  });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
