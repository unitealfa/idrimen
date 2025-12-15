"use client";

import { useEffect, useRef, useState } from 'react';
import { createChart, ColorType, IChartApi, ISeriesApi, Time, CandlestickData } from 'lightweight-charts';

// Data types
interface Candle {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
}

type CandlesData = Record<string, Candle[]>;

// Icons
const ClockIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-clock"><circle cx="12" cy="12" r="10" /><polyline points="12 6 12 12 16 14" /></svg>
);

const ActivityIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-activity"><path d="M22 12h-4l-3 9L9 3l-3 9H2" /></svg>
);

const TrendingUpIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-trending-up"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18" /><polyline points="17 6 23 6 23 12" /></svg>
);

const TrendingDownIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-trending-down"><polyline points="23 18 13.5 8.5 8.5 13.5 1 6" /><polyline points="17 18 23 18 23 12" /></svg>
);

// Components
function Header() {
  const [time, setTime] = useState("");

  useEffect(() => {
    const timer = setInterval(() => {
      setTime(new Date().toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit', second: '2-digit' }));
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  return (
    <header className="header px-6 py-4 flex items-center justify-between">
      <div className="flex items-center gap-4">
        <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
          <span className="text-3xl">⚡</span>
          <span className="bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent logo-animated">
            Forex Pro
          </span>
        </h1>
        <div className="hidden md:flex ml-8 gap-4 text-sm font-medium text-gray-400">
          <span className="hover:text-white cursor-pointer transition-colors">Dashboard</span>
          <span className="hover:text-white cursor-pointer transition-colors">Markets</span>
          <span className="hover:text-white cursor-pointer transition-colors">News</span>
        </div>
      </div>

      <div className="flex items-center gap-6">
        <div className="flex items-center gap-2 text-gray-400 text-sm font-mono">
          <ClockIcon />
          <span>{time}</span>
        </div>
        <div className="live-indicator">
          <span className="live-dot"></span>
          LIVE MARKET
        </div>
      </div>
    </header>
  );
}

function StatCard({ label, value, trend }: { label: string, value: string, trend?: string }) {
  return (
    <div className="stat-card">
      <div className="text-gray-400 text-xs uppercase tracking-wider font-semibold">{label}</div>
      <div className="flex items-baseline gap-2 mt-1">
        <div className="text-2xl font-bold font-mono text-white">{value}</div>
        {trend && <div className="text-xs text-emerald-400">{trend}</div>}
      </div>
    </div>
  )
}


// Chart Component
function ChartCard({ symbol, data }: { symbol: string, data: Candle[] }) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);

  const lastCandle = data[data.length - 1];
  const prevCandle = data[data.length - 2] || lastCandle;
  const isBullish = lastCandle.close >= prevCandle.close;

  // Calculate percentage change
  const priceChange = lastCandle.close - prevCandle.close;
  const percentChange = ((priceChange / prevCandle.close) * 100).toFixed(4);
  const changeColor = isBullish ? 'text-emerald-400' : 'text-red-400';
  const ChartIcon = isBullish ? TrendingUpIcon : TrendingDownIcon;
  const bgClass = isBullish ? 'price-badge-up' : 'price-badge-down';
  // Format price to 5 decimals usually for forex
  const currentPrice = lastCandle.close.toFixed(5);


  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#64748b',
        fontFamily: "'JetBrains Mono', monospace",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: 'rgba(255, 255, 255, 0.05)' },
        horzLines: { color: 'rgba(255, 255, 255, 0.05)' },
      },
      width: chartContainerRef.current.clientWidth,
      height: 280,
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
        borderColor: 'rgba(255, 255, 255, 0.1)',
      },
      rightPriceScale: {
        borderColor: 'rgba(255, 255, 255, 0.1)',
      },
      crosshair: {
        mode: 0, // Normal
        vertLine: {
          width: 1,
          color: 'rgba(255, 255, 255, 0.4)',
          style: 3,
          labelBackgroundColor: '#6366f1',
        },
        horzLine: {
          width: 1,
          color: 'rgba(255, 255, 255, 0.4)',
          style: 3,
          labelBackgroundColor: '#6366f1',
        },
      }
    });

    chartRef.current = chart;

    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#10b981',
      downColor: '#ef4444',
      borderVisible: false,
      wickUpColor: '#10b981',
      wickDownColor: '#ef4444',
    });

    seriesRef.current = candlestickSeries;
    candlestickSeries.setData(data as unknown as CandlestickData<Time>[]);

    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, []);

  useEffect(() => {
    if (seriesRef.current && data.length > 0) {
      seriesRef.current.setData(data as unknown as CandlestickData<Time>[]);
    }
  }, [data]);

  return (
    <div className={`glass-card p-5 flex flex-col gap-4 ${isBullish ? 'chart-card-bullish' : 'chart-card-bearish'} grid-item`}>
      <div className="flex justify-between items-start">
        <div>
          <h2 className="text-lg font-bold text-white tracking-wide">{symbol}</h2>
          <div className="flex items-center gap-2 mt-1">
            <span className="text-2xl font-mono font-medium text-white">{currentPrice}</span>
          </div>
        </div>
        <div className={`flex flex-col items-end ${changeColor}`}>
          <span className={`price-badge ${bgClass}`}>
            {Number(percentChange) > 0 ? '+' : ''}{percentChange}%
            <ChartIcon />
          </span>
        </div>
      </div>

      <div ref={chartContainerRef} className="w-full h-[280px] rounded-lg overflow-hidden" />
    </div>
  );
}

function LoadingSkeleton() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 animate-pulse">
      {[1, 2, 3, 4, 5, 6].map(i => (
        <div key={i} className="h-[350px] bg-white/5 rounded-2xl border border-white/10" />
      ))}
    </div>
  )
}

export default function Home() {
  const [candlesData, setCandlesData] = useState<CandlesData>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch('/api/tick');
        const json = await res.json();
        setCandlesData(json);
        if (loading && Object.keys(json).length > 0) setLoading(false);
      } catch (e) {
        console.error("Failed to fetch", e);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 1000);
    return () => clearInterval(interval);
  }, []);

  const symbols = Object.keys(candlesData).sort();
  const activePairs = symbols.length;

  return (
    <div className="min-h-screen pb-20">
      <Header />

      <main className="container mx-auto px-6 py-8">

        {/* Top Stats Row */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10 animate-fade-in">
          <StatCard label="Active Pairs" value={activePairs.toString()} trend="Online" />
          <StatCard label="Market Status" value="OPEN" trend="24/5" />
          <StatCard label="Update Rate" value="1000ms" trend="Live Tick" />
        </div>

        {symbols.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-20 text-center">
            {loading ? (
              <LoadingSkeleton />
            ) : (
              <>
                <div className="w-16 h-16 rounded-full border-4 border-blue-500/30 border-t-blue-500 animate-spin mb-6"></div>
                <h2 className="text-2xl font-bold text-white mb-2">Connecting to Feed...</h2>
                <p className="text-gray-400">Waiting for live tick data from Python bridge</p>
              </>
            )}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {symbols.map(symbol => (
              <ChartCard key={symbol} symbol={symbol} data={candlesData[symbol]} />
            ))}
          </div>
        )}
      </main>

      <footer className="fixed bottom-0 w-full py-3 bg-black/80 backdrop-blur-md border-t border-white/5 text-center text-xs text-gray-500">
        <p>Forex Pro Dashboard v2.0 • Powered by Next.js & Python • Data provided for demonstration</p>
      </footer>
    </div>
  );
}
