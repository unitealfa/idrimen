
import { NextResponse } from 'next/server';

interface Candle {
  time: number; // Unix timestamp in seconds
  open: number;
  high: number;
  low: number;
  close: number;
}

// In-memory storage: { "EURUSD": [Candle, Candle, ...], ... }
// globalThis is used to persist across hot-reloads in dev, 
// though in Vercel Serverless it might reset on cold starts.
const globalStore = globalThis as unknown as { candles: Record<string, Candle[]> };
if (!globalStore.candles) {
  globalStore.candles = {};
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { symbol, price, timestamp } = body; 
    // timestamp expected to be string ISO or similar, we convert to unix seconds for charts

    if (!symbol || !price) {
      return NextResponse.json({ error: 'Missing symbol or price' }, { status: 400 });
    }

    const priceFloat = parseFloat(price);
    const date = timestamp ? new Date(timestamp) : new Date();
    // Round to nearest minute for 1-minute candles
    const timeSeconds = Math.floor(date.getTime() / 1000);
    const candleTime = timeSeconds - (timeSeconds % 60);

    if (!globalStore.candles[symbol]) {
        globalStore.candles[symbol] = [];
    }
    
    const series = globalStore.candles[symbol];
    const lastCandle = series.length > 0 ? series[series.length - 1] : null;

    if (lastCandle && lastCandle.time === candleTime) {
      // Update existing candle
      lastCandle.close = priceFloat;
      lastCandle.high = Math.max(lastCandle.high, priceFloat);
      lastCandle.low = Math.min(lastCandle.low, priceFloat);
    } else {
      // New candle
      const newCandle: Candle = {
        time: candleTime,
        open: lastCandle ? lastCandle.close : priceFloat,
        high: priceFloat,
        low: priceFloat,
        close: priceFloat,
      };
      series.push(newCandle);
      // Keep only last 1000 candles to avoid memory overflow
      if (series.length > 1000) {
          series.shift();
      }
    }

    return NextResponse.json({ success: true, count: series.length });
  } catch (error) {
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}

export async function GET() {
  return NextResponse.json(globalStore.candles);
}
