const pairSelect = document.getElementById("pair-select");
const realtimeBody = document.getElementById("realtime-body");
const priceChartEl = document.getElementById("priceChart");
const clockEl = document.getElementById("clock");

const fxPairs = [
  "EUR/USD",
  "USD/JPY",
  "GBP/USD",
  "AUD/USD",
  "USD/CAD",
  "USD/CHF",
  "NZD/USD",
  "EUR/GBP",
  "EUR/JPY",
  "EUR/CHF",
  "EUR/AUD",
  "EUR/CAD",
  "GBP/JPY",
  "AUD/JPY",
  "CHF/JPY",
  "USD/CNY",
  "USD/HKD",
  "USD/SGD",
  "USD/KRW",
  "USD/INR",
  "USD/MXN",
  "USD/ZAR",
];

let ws = null;
let currentPair = fxPairs[0];
let lcChart = null;
let candleSeries = null;
let candleMap = new Map(); // key: bucketSec -> candle
let candleData = [];
let currentPairNormalized = normalizeSymbol(currentPair);
const TZ_OFFSET_MS = 60 * 60 * 1000; // UTC+1
const TIME_ZONE = "Europe/Paris";
let clockInterval = null;

function normalizeSymbol(raw) {
  const trimmed = (raw || "").trim().toUpperCase();
  if (!trimmed) return "EUR/USD";
  if (trimmed.includes("/")) {
    return `OANDA:${trimmed.replace(/\//g, "_")}`;
  }
  return trimmed;
}

function ensureChart() {
  if (!priceChartEl || !window.LightweightCharts) {
    console.error("LightweightCharts non charge.");
    return;
  }
  if (!lcChart) {
    lcChart = LightweightCharts.createChart(priceChartEl, {
      layout: { background: { color: "#0f172a" }, textColor: "#e7ecf5" },
      grid: {
        vertLines: { color: "#1f2a45" },
        horzLines: { color: "#1f2a45" },
      },
      rightPriceScale: { borderColor: "#1f2a45" },
      timeScale: {
        borderColor: "#1f2a45",
        timeVisible: true,
        secondsVisible: false,
      },
      crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    });
    candleSeries = lcChart.addCandlestickSeries({
      upColor: "#40c463",
      borderUpColor: "#40c463",
      wickUpColor: "#40c463",
      downColor: "#e63946",
      borderDownColor: "#e63946",
      wickDownColor: "#e63946",
    });
  }
}

function populateSelect() {
  if (!pairSelect) return;
  fxPairs.forEach((pair) => {
    const opt = document.createElement("option");
    opt.value = pair;
    opt.textContent = pair;
    pairSelect.appendChild(opt);
  });
  pairSelect.value = currentPair;
  startClock();
}

function startClock() {
  if (!clockEl) return;
  if (clockInterval) clearInterval(clockInterval);
  const update = () => {
    const now = new Date();
    clockEl.textContent = now.toLocaleString("fr-FR", {
      timeZone: TIME_ZONE,
      hour12: false,
    });
  };
  update();
  clockInterval = setInterval(update, 1000);
}

function resetChartForPair() {
  candleMap.clear();
  candleData = [];
  ensureChart();
  if (candleSeries) {
    candleSeries.setData([]);
  }
}

function updateChartWithTrade(trade) {
  const selectedSymbol = currentPairNormalized;
  if (trade.s && trade.s.toUpperCase() !== selectedSymbol) return;

  const price = trade.p;
  const bucketSec = Math.floor((trade.t + TZ_OFFSET_MS) / 60000) * 60; // 1 minute UTC+1
  ensureChart();
  if (!candleSeries) return;

  let candle = candleMap.get(bucketSec);
  if (!candle) {
    candle = { time: bucketSec, open: price, high: price, low: price, close: price };
    candleMap.set(bucketSec, candle);
    candleData.push(candle);
    candleData.sort((a, b) => a.time - b.time);
    if (candleData.length > 300) {
      const removed = candleData.shift();
      if (removed) candleMap.delete(removed.time);
    }
  } else {
    candle.high = Math.max(candle.high, price);
    candle.low = Math.min(candle.low, price);
    candle.close = price;
  }
  candleSeries.setData(candleData);
}

function addRealtimeRow(trade) {
  const selectedSymbol = currentPairNormalized;
  if (trade.s && trade.s.toUpperCase() !== selectedSymbol) return;

  const tr = document.createElement("tr");
  const timeTd = document.createElement("td");
  const symTd = document.createElement("td");
  const priceTd = document.createElement("td");
  const volTd = document.createElement("td");

  const date = new Date(trade.t);
  timeTd.textContent = date.toLocaleTimeString("fr-FR", {
    timeZone: TIME_ZONE,
    hour12: false,
  });
  symTd.textContent = pairSelect.value || currentPair;
  priceTd.textContent = trade.p;
  volTd.textContent = trade.v;

  tr.appendChild(timeTd);
  tr.appendChild(symTd);
  tr.appendChild(priceTd);
  tr.appendChild(volTd);

  realtimeBody.insertBefore(tr, realtimeBody.firstChild);
  while (realtimeBody.rows.length > 50) {
    realtimeBody.deleteRow(realtimeBody.rows.length - 1);
  }
}

function connectWebSocket(pair) {
  const symbol = normalizeSymbol(pair);
  currentPairNormalized = symbol.toUpperCase();
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const wsUrl = `${protocol}://${window.location.host}`;

  if (ws) {
    ws.onclose = null;
    ws.close();
  }

  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    console.log("WS navigateur connecte au serveur");
    ws.send(JSON.stringify({ type: "set-symbol", symbol }));
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      if (msg.type === "trade" && Array.isArray(msg.data)) {
        msg.data.forEach((t) => {
          addRealtimeRow(t);
          updateChartWithTrade(t);
        });
      }
    } catch (e) {
      console.error("Message WS invalide:", e);
    }
  };

  ws.onclose = () => {
    console.log("WS ferme. Reconnexion dans 3s...");
    setTimeout(() => connectWebSocket(currentPair), 3000);
  };

  ws.onerror = (err) => {
    console.error("WS error:", err);
  };
}

function onPairChange() {
  currentPair = pairSelect.value || fxPairs[0];
  currentPairNormalized = normalizeSymbol(currentPair).toUpperCase();
  realtimeBody.innerHTML = "";
  resetChartForPair();
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(
      JSON.stringify({ type: "set-symbol", symbol: normalizeSymbol(currentPair) })
    );
  } else {
    connectWebSocket(currentPair);
  }
}

function init() {
  populateSelect();
  resetChartForPair();
  connectWebSocket(currentPair);
  pairSelect.addEventListener("change", onPairChange);
}

init();
