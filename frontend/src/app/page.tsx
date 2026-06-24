"use client";

import { useEffect, useState, useRef } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Shield, ArrowRight, Camera, Brain, Zap, Eye, BarChart3, ChevronDown, Menu, CheckCircle, AlertTriangle, TrendingUp, Layers } from "lucide-react";

function useCountUp(end: number, duration: number, trigger: boolean) {
  const [count, setCount] = useState(0);
  useEffect(() => {
    if (!trigger) return;
    let start = 0;
    const increment = end / (duration / 16);
    const timer = setInterval(() => {
      start += increment;
      if (start >= end) { setCount(end); clearInterval(timer); }
      else setCount(Math.floor(start));
    }, 16);
    return () => clearInterval(timer);
  }, [end, duration, trigger]);
  return count;
}

function useInView(threshold = 0.2) {
  const ref = useRef<HTMLDivElement>(null);
  const [inView, setInView] = useState(false);
  useEffect(() => {
    const obs = new IntersectionObserver(([e]) => { if (e.isIntersecting) setInView(true); }, { threshold });
    if (ref.current) obs.observe(ref.current);
    return () => obs.disconnect();
  }, []);
  return { ref, inView };
}

export default function LandingPage() {
  const router = useRouter();
  const [menuOpen, setMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const statsRef = useRef<HTMLDivElement>(null);
  const [statsInView, setStatsInView] = useState(false);

  const { ref: problemRef, inView: problemInView } = useInView();
  const { ref: yoloRef, inView: yoloInView } = useInView();
  const { ref: cnnRef, inView: cnnInView } = useInView();
  const { ref: xgbRef, inView: xgbInView } = useInView();
  const { ref: shapRef, inView: shapInView } = useInView();
  const { ref: fusionRef, inView: fusionInView } = useInView();

  useEffect(() => {
    const token = localStorage.getItem("access_token");
    const role = localStorage.getItem("user_role")?.toLowerCase();
    if (token) {
      if (role === "admin") router.push("/admin");
      else if (role === "surveyor") router.push("/surveyor");
      else router.push("/customer");
    }
  }, [router]);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    const obs = new IntersectionObserver(([e]) => setStatsInView(e.isIntersecting), { threshold: 0.3 });
    if (statsRef.current) obs.observe(statsRef.current);
    return () => obs.disconnect();
  }, []);

  const claims    = useCountUp(10000, 1500, statsInView);
  const accuracy  = useCountUp(96, 1200, statsInView);
  const saved     = useCountUp(45000, 1800, statsInView);
  const speed     = useCountUp(10, 1000, statsInView);

  return (
    <div className="bg-[#0a0a0a] text-white min-h-screen w-full" style={{ fontFamily: "'DM Sans', 'Helvetica Neue', sans-serif" }}>

      {/* ── NAV ── */}
      <header className={`fixed top-0 w-full z-50 transition-all duration-300 ${scrolled ? "bg-[#0a0a0a]/95 border-b border-white/10 backdrop-blur-md" : ""}`}>
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 bg-white rounded-lg flex items-center justify-center">
              <Shield className="h-4 w-4 text-black" />
            </div>
            <span className="font-bold text-white text-lg tracking-tight">SmartClaim AI</span>
          </div>
          <nav className="hidden md:flex items-center gap-8 text-sm text-white/60">
            <a href="#problem" className="hover:text-white transition-colors">Problem</a>
            <a href="#how" className="hover:text-white transition-colors">How It Works</a>
            <a href="#models" className="hover:text-white transition-colors">AI Models</a>
            <a href="#explainability" className="hover:text-white transition-colors">Explainability</a>
            <a href="#impact" className="hover:text-white transition-colors">Impact</a>
          </nav>
          <div className="hidden md:flex items-center gap-3">
            <Link href="/login" className="text-sm text-white/60 hover:text-white transition-colors px-4 py-2">Login</Link>
            <Link href="/register" className="text-sm bg-white text-black font-semibold px-5 py-2 rounded-lg hover:bg-gray-100 transition-colors">Get Started</Link>
          </div>
          <button className="md:hidden text-white" onClick={() => setMenuOpen(!menuOpen)}>
            {menuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </button>
        </div>
        {menuOpen && (
          <div className="md:hidden bg-[#0a0a0a] border-t border-white/10 px-6 py-6 space-y-4">
            {["problem","how","models","explainability","impact"].map(s => (
              <a key={s} href={`#${s}`} onClick={() => setMenuOpen(false)} className="block text-white/70 hover:text-white capitalize">{s}</a>
            ))}
            <div className="pt-4 flex gap-3">
              <Link href="/login" className="flex-1 text-center text-sm border border-white/20 text-white py-2 rounded-lg">Login</Link>
              <Link href="/register" className="flex-1 text-center text-sm bg-white text-black font-semibold py-2 rounded-lg">Get Started</Link>
            </div>
          </div>
        )}
      </header>

      {/* ── HERO ── */}
      <section className="relative min-h-screen flex flex-col items-center justify-center px-6 text-center overflow-hidden">
        {/* Background grid */}
        <div className="absolute inset-0" style={{
          backgroundImage: "linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px)",
          backgroundSize: "60px 60px"
        }} />
        {/* Glow */}
        <div className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full" style={{ background: "radial-gradient(circle, rgba(255,255,255,0.04) 0%, transparent 70%)" }} />

        <div className="relative max-w-4xl mx-auto">
          <div className="inline-flex items-center gap-2 border border-white/15 rounded-full px-4 py-1.5 text-xs text-white/50 mb-8 bg-white/5">
            <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
            AI-Powered · Explainable · Automated
          </div>

          <h1 className="text-5xl md:text-7xl lg:text-8xl font-black tracking-tighter leading-[0.9] mb-6">
            Insurance Claims<br />
            <span style={{ WebkitTextStroke: "1px rgba(255,255,255,0.3)", color: "transparent" }}>
              Decided by AI.
            </span><br />
            <span className="text-white">Explained to Humans.</span>
          </h1>

          <p className="text-white/50 text-lg md:text-xl max-w-2xl mx-auto mb-10 leading-relaxed">
            SmartClaim AI uses YOLO, CNN, and XGBoost to detect damage, calculate amounts, and score fraud — then explains every decision with SHAP.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link href="/register" className="inline-flex items-center gap-2 bg-white text-black font-bold px-8 py-4 rounded-xl hover:bg-gray-100 transition-all text-sm">
              File a Claim <ArrowRight className="h-4 w-4" />
            </Link>
            <a href="#problem" className="inline-flex items-center gap-2 border border-white/20 text-white/70 hover:text-white hover:border-white/40 font-medium px-8 py-4 rounded-xl transition-all text-sm">
              See How It Works <ChevronDown className="h-4 w-4" />
            </a>
          </div>
        </div>

        {/* Scroll indicator */}
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 text-white/30 text-xs animate-bounce">
          <span>scroll to explore</span>
          <ChevronDown className="h-4 w-4" />
        </div>
      </section>

      {/* ── PROBLEM ── */}
      <section id="problem" className="py-32 px-6">
        <div ref={problemRef} className="max-w-6xl mx-auto">
          <div className={`transition-all duration-700 ${problemInView ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8"}`}>
            <div className="text-xs font-bold tracking-widest text-white/30 uppercase mb-4">The Problem</div>
            <h2 className="text-4xl md:text-6xl font-black tracking-tight mb-16 leading-tight">
              ₹45,000 Crores lost<br />
              <span className="text-white/30">to fraud every year.</span>
            </h2>
          </div>

          <div className="grid md:grid-cols-3 gap-px bg-white/5 rounded-2xl overflow-hidden">
            {[
              { num: "3 Weeks", label: "Average claim settlement time", sub: "Traditional process" },
              { num: "Opaque", label: "Decisions with no explanation", sub: "Adjuster says no. No reason given." },
              { num: "Biased", label: "Human subjectivity in assessment", sub: "Same damage, different adjusters, different amounts." },
            ].map(({ num, label, sub }) => (
              <div key={num} className="bg-[#111] p-8 md:p-10">
                <p className="text-3xl md:text-4xl font-black text-red-400 mb-3">{num}</p>
                <p className="text-white font-semibold mb-2">{label}</p>
                <p className="text-white/40 text-sm">{sub}</p>
              </div>
            ))}
          </div>

          <div className="mt-16 p-8 md:p-12 rounded-2xl border border-white/10 bg-white/[0.02]">
            <p className="text-2xl md:text-3xl font-bold text-white/80 leading-relaxed">
              "The system is broken. Claims take weeks. Fraud slips through. Honest customers are penalised. There is no transparency."
            </p>
            <p className="text-white/30 mt-4 text-sm">— The problem SmartClaim AI solves</p>
          </div>
        </div>
      </section>

      {/* ── HOW IT WORKS ── */}
      <section id="how" className="py-32 px-6 border-t border-white/5">
        <div className="max-w-6xl mx-auto">
          <div className="text-xs font-bold tracking-widest text-white/30 uppercase mb-4">The Solution</div>
          <h2 className="text-4xl md:text-6xl font-black tracking-tight mb-6">Photo in. Decision out.<br /><span className="text-white/30">In under 10 seconds.</span></h2>
          <p className="text-white/50 text-lg mb-20 max-w-2xl">Three AI models run simultaneously the moment photos are uploaded. No waiting. No subjectivity.</p>

          {/* Pipeline visual */}
          <div className="relative">
            <div className="hidden md:block absolute top-12 left-[10%] right-[10%] h-px bg-gradient-to-r from-transparent via-white/20 to-transparent" />
            <div className="grid md:grid-cols-4 gap-6">
              {[
                { step: "01", icon: Camera, label: "Upload Photos", desc: "Customer uploads damage photos from phone or desktop. Multiple angles supported.", color: "text-blue-400" },
                { step: "02", icon: Brain, label: "3-Model AI Pipeline", desc: "YOLO + CNN + XGBoost run in parallel. Damage detected, severity scored, fraud assessed.", color: "text-violet-400" },
                { step: "03", icon: Eye, label: "SHAP Explanation", desc: "Every decision explained. Which features drove the fraud score. Why the amount is what it is.", color: "text-emerald-400" },
                { step: "04", icon: CheckCircle, label: "Instant Decision", desc: "Claim verified, flagged, or rejected — with a full PDF report the customer can download.", color: "text-amber-400" },
              ].map(({ step, icon: Icon, label, desc, color }) => (
                <div key={step} className="relative bg-[#111] border border-white/8 rounded-2xl p-6 hover:border-white/20 transition-colors">
                  <div className="text-xs font-black text-white/20 mb-4">{step}</div>
                  <Icon className={`h-6 w-6 ${color} mb-4`} />
                  <p className="font-bold text-white mb-2">{label}</p>
                  <p className="text-white/40 text-sm leading-relaxed">{desc}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ── MODEL 1: YOLO ── */}
      <section id="models" className="py-32 px-6 border-t border-white/5">
        <div ref={yoloRef} className="max-w-6xl mx-auto">
          <div className={`transition-all duration-700 ${yoloInView ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8"}`}>
            <div className="grid md:grid-cols-2 gap-16 items-center">
              <div>
                <div className="inline-flex items-center gap-2 text-xs font-bold tracking-widest text-blue-400 uppercase mb-6 border border-blue-400/30 rounded-full px-4 py-1.5">
                  Model 1 — Computer Vision
                </div>
                <h2 className="text-4xl md:text-5xl font-black tracking-tight mb-6 leading-tight">
                  YOLOv8<br /><span className="text-white/30">sees what the eye misses.</span>
                </h2>
                <p className="text-white/50 text-lg leading-relaxed mb-8">
                  You Only Look Once — real-time object detection trained on thousands of vehicle damage images. Identifies every damaged part in milliseconds.
                </p>
                <div className="space-y-4">
                  {[
                    "Detects bumpers, hoods, doors, lights, windshields — individually",
                    "Draws bounding boxes around every damage region",
                    "Each detection has a confidence score",
                    "Works across multiple photos simultaneously",
                  ].map(item => (
                    <div key={item} className="flex items-start gap-3">
                      <div className="w-5 h-5 rounded-full bg-blue-400/20 border border-blue-400/40 flex items-center justify-center flex-shrink-0 mt-0.5">
                        <div className="w-1.5 h-1.5 rounded-full bg-blue-400" />
                      </div>
                      <p className="text-white/60 text-sm">{item}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* YOLO visual */}
              <div className="bg-[#111] border border-white/10 rounded-2xl overflow-hidden">
                <div className="p-4 border-b border-white/5 flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-red-500/60" />
                  <div className="w-2 h-2 rounded-full bg-amber-500/60" />
                  <div className="w-2 h-2 rounded-full bg-green-500/60" />
                  <span className="text-xs text-white/30 ml-2">YOLO Detection Output</span>
                </div>
                <div className="p-6 space-y-3">
                  {[
                    { part: "Front Bumper", conf: 94, damage: "Dent + Crack", color: "bg-red-500" },
                    { part: "Hood",         conf: 87, damage: "Paint Scratch", color: "bg-orange-500" },
                    { part: "Headlight L",  conf: 91, damage: "Broken",       color: "bg-red-600" },
                    { part: "Fender",       conf: 78, damage: "Dent",         color: "bg-amber-500" },
                    { part: "Windshield",   conf: 62, damage: "Minor Crack",  color: "bg-yellow-500" },
                  ].map(({ part, conf, damage, color }) => (
                    <div key={part} className="flex items-center gap-3 p-3 bg-white/[0.03] rounded-lg border border-white/5">
                      <div className={`w-2 h-8 rounded-full ${color} opacity-80 flex-shrink-0`} />
                      <div className="flex-1">
                        <div className="flex items-center justify-between mb-1">
                          <p className="text-sm font-semibold text-white">{part}</p>
                          <span className="text-xs text-white/40 font-mono">{conf}%</span>
                        </div>
                        <p className="text-xs text-white/40">{damage}</p>
                        <div className="mt-1.5 h-1 bg-white/10 rounded-full overflow-hidden">
                          <div className={`h-full ${color} rounded-full opacity-70`} style={{ width: `${conf}%` }} />
                        </div>
                      </div>
                    </div>
                  ))}
                  <div className="pt-2 border-t border-white/5 flex justify-between text-xs text-white/30">
                    <span>5 parts detected</span>
                    <span className="text-blue-400">Base amount: ₹52,400</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── MODEL 2: CNN ── */}
      <section className="py-32 px-6 border-t border-white/5 bg-[#0d0d0d]">
        <div ref={cnnRef} className="max-w-6xl mx-auto">
          <div className={`transition-all duration-700 ${cnnInView ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8"}`}>
            <div className="grid md:grid-cols-2 gap-16 items-center">
              {/* CNN visual */}
              <div className="order-2 md:order-1 bg-[#111] border border-white/10 rounded-2xl overflow-hidden">
                <div className="p-4 border-b border-white/5 flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-red-500/60" />
                  <div className="w-2 h-2 rounded-full bg-amber-500/60" />
                  <div className="w-2 h-2 rounded-full bg-green-500/60" />
                  <span className="text-xs text-white/30 ml-2">CNN Damage Analysis</span>
                </div>
                <div className="p-6">
                  <div className="space-y-3 mb-6">
                    {[
                      { label: "Image 1", pct: 34, severity: "MODERATE", c: "bg-orange-500" },
                      { label: "Image 2", pct: 67, severity: "SEVERE",   c: "bg-red-500" },
                      { label: "Image 3", pct: 18, severity: "MINOR",    c: "bg-yellow-500" },
                      { label: "Image 4", pct: 52, severity: "MODERATE", c: "bg-orange-400" },
                    ].map(({ label, pct, severity, c }) => (
                      <div key={label} className="p-3 bg-white/[0.03] rounded-lg border border-white/5">
                        <div className="flex justify-between items-center mb-2">
                          <span className="text-sm text-white/70">{label}</span>
                          <span className={`text-xs font-bold px-2 py-0.5 rounded ${c} text-white bg-opacity-20`} style={{ backgroundColor: "rgba(255,255,255,0.1)" }}>{severity}</span>
                        </div>
                        <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                          <div className={`h-full ${c} rounded-full`} style={{ width: `${pct}%` }} />
                        </div>
                        <div className="flex justify-between mt-1">
                          <span className="text-xs text-white/30">Damage area</span>
                          <span className="text-xs text-white/50 font-mono">{pct}%</span>
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="p-4 bg-violet-500/10 border border-violet-500/20 rounded-xl">
                    <p className="text-xs text-violet-400 font-semibold mb-1">CNN Multiplier Applied</p>
                    <p className="text-2xl font-black text-white">1.67×</p>
                    <p className="text-xs text-white/40 mt-1">Base ₹52,400 → Final ₹87,500</p>
                  </div>
                </div>
              </div>

              <div className="order-1 md:order-2">
                <div className="inline-flex items-center gap-2 text-xs font-bold tracking-widest text-violet-400 uppercase mb-6 border border-violet-400/30 rounded-full px-4 py-1.5">
                  Model 2 — Damage Severity
                </div>
                <h2 className="text-4xl md:text-5xl font-black tracking-tight mb-6 leading-tight">
                  CNN measures<br /><span className="text-white/30">how bad it really is.</span>
                </h2>
                <p className="text-white/50 text-lg leading-relaxed mb-8">
                  YOLO tells us what is damaged. The Convolutional Neural Network tells us how severely. It reads pixel-level damage patterns across the entire image.
                </p>
                <div className="space-y-4">
                  {[
                    "Scans every pixel for damage patterns",
                    "Assigns damage percentage per image",
                    "Classifies severity: Minor, Moderate, Severe",
                    "Generates a multiplier applied to YOLO base amount",
                    "Produces annotated overlay images as evidence",
                  ].map(item => (
                    <div key={item} className="flex items-start gap-3">
                      <div className="w-5 h-5 rounded-full bg-violet-400/20 border border-violet-400/40 flex items-center justify-center flex-shrink-0 mt-0.5">
                        <div className="w-1.5 h-1.5 rounded-full bg-violet-400" />
                      </div>
                      <p className="text-white/60 text-sm">{item}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── MODEL 3: XGBOOST ── */}
      <section className="py-32 px-6 border-t border-white/5">
        <div ref={xgbRef} className="max-w-6xl mx-auto">
          <div className={`transition-all duration-700 ${xgbInView ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8"}`}>
            <div className="grid md:grid-cols-2 gap-16 items-center">
              <div>
                <div className="inline-flex items-center gap-2 text-xs font-bold tracking-widest text-red-400 uppercase mb-6 border border-red-400/30 rounded-full px-4 py-1.5">
                  Model 3 — Fraud Detection
                </div>
                <h2 className="text-4xl md:text-5xl font-black tracking-tight mb-6 leading-tight">
                  XGBoost scores<br /><span className="text-white/30">73 fraud signals.</span>
                </h2>
                <p className="text-white/50 text-lg leading-relaxed mb-8">
                  Gradient boosted decision trees analyse behavioural, temporal, and policy-level patterns to generate a fraud probability score in real time.
                </p>
                <div className="grid grid-cols-2 gap-4">
                  {[
                    { signal: "Days since policy start", flag: true },
                    { signal: "Previous claims count", flag: true },
                    { signal: "Police report filed", flag: false },
                    { signal: "Witness present", flag: false },
                    { signal: "Agent type", flag: true },
                    { signal: "Accident area", flag: false },
                    { signal: "Claim day of week", flag: false },
                    { signal: "Address change recency", flag: true },
                  ].map(({ signal, flag }) => (
                    <div key={signal} className={`p-3 rounded-lg border text-xs flex items-center gap-2 ${flag ? "border-red-500/30 bg-red-500/5" : "border-white/10 bg-white/[0.02]"}`}>
                      <div className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${flag ? "bg-red-400" : "bg-green-400"}`} />
                      <span className={flag ? "text-red-300" : "text-white/40"}>{signal}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* XGBoost visual */}
              <div className="bg-[#111] border border-white/10 rounded-2xl overflow-hidden">
                <div className="p-4 border-b border-white/5 flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-red-500/60" />
                  <div className="w-2 h-2 rounded-full bg-amber-500/60" />
                  <div className="w-2 h-2 rounded-full bg-green-500/60" />
                  <span className="text-xs text-white/30 ml-2">XGBoost Fraud Score</span>
                </div>
                <div className="p-6">
                  {/* Gauge */}
                  <div className="flex flex-col items-center mb-6">
                    <div className="relative w-40 h-20 overflow-hidden mb-4">
                      <div className="absolute inset-0 rounded-t-full border-8 border-white/5" />
                      <div className="absolute inset-0 rounded-t-full border-8 border-transparent border-t-red-500 border-r-red-500" style={{ transform: "rotate(25deg)", transformOrigin: "center bottom" }} />
                      <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-1 h-16 bg-white rounded-full origin-bottom" style={{ transform: "rotate(30deg)" }} />
                      <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-4 h-4 rounded-full bg-[#111] border-2 border-white/30" />
                    </div>
                    <p className="text-5xl font-black text-white">73<span className="text-2xl text-white/40">%</span></p>
                    <p className="text-xs text-white/30 mt-1">Fraud Probability</p>
                    <div className="mt-3 px-4 py-1.5 rounded-full bg-red-500/20 border border-red-500/30 text-red-400 text-xs font-bold">HIGH RISK</div>
                  </div>
                  {/* Score breakdown */}
                  <div className="space-y-2 border-t border-white/5 pt-4">
                    {[
                      { label: "Tabular (XGBoost)",  val: 71, c: "bg-red-500" },
                      { label: "Image (CNN)",         val: 68, c: "bg-orange-500" },
                      { label: "Fusion Score",        val: 73, c: "bg-red-600" },
                    ].map(({ label, val, c }) => (
                      <div key={label} className="flex items-center gap-3">
                        <span className="text-xs text-white/40 w-36 flex-shrink-0">{label}</span>
                        <div className="flex-1 h-2 bg-white/10 rounded-full overflow-hidden">
                          <div className={`h-full ${c} rounded-full`} style={{ width: `${val}%` }} />
                        </div>
                        <span className="text-xs font-mono text-white/50 w-8 text-right">{val}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── EXPLAINABILITY ── */}
      <section id="explainability" className="py-32 px-6 border-t border-white/5 bg-[#0d0d0d]">
        <div ref={shapRef} className="max-w-6xl mx-auto">
          <div className={`transition-all duration-700 ${shapInView ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8"}`}>
            <div className="text-center mb-16">
              <div className="inline-flex items-center gap-2 text-xs font-bold tracking-widest text-emerald-400 uppercase mb-6 border border-emerald-400/30 rounded-full px-4 py-1.5">
                What Makes Us Different
              </div>
              <h2 className="text-4xl md:text-6xl font-black tracking-tight mb-6">
                Not just a score.<br /><span className="text-white/30">A reason.</span>
              </h2>
              <p className="text-white/50 text-xl max-w-2xl mx-auto">
                SHAP — SHapley Additive Explanations — reveals exactly which factors drove the AI's fraud decision. Based on Nobel-winning game theory.
              </p>
            </div>

            <div className="grid md:grid-cols-2 gap-8 mb-12">
              {/* Without XAI */}
              <div className="bg-[#111] border border-red-500/20 rounded-2xl p-8">
                <div className="flex items-center gap-2 mb-6">
                  <X className="h-4 w-4 text-red-400" />
                  <p className="text-sm font-bold text-red-400">Without Explainability</p>
                </div>
                <div className="space-y-3">
                  <div className="p-4 bg-white/[0.03] rounded-xl border border-white/5">
                    <p className="text-white/50 text-sm">AI Output</p>
                    <p className="text-3xl font-black text-white mt-1">73% Fraud</p>
                  </div>
                  <div className="p-4 bg-white/[0.03] rounded-xl border border-white/5">
                    <p className="text-white/50 text-sm">Why?</p>
                    <p className="text-2xl font-black text-white/20 mt-1">¯\_(ツ)_/¯</p>
                  </div>
                  <p className="text-white/30 text-xs pt-2">Admin makes decision blindly. Customer has no recourse. Legally indefensible.</p>
                </div>
              </div>

              {/* With SHAP */}
              <div className="bg-[#111] border border-emerald-500/20 rounded-2xl p-8">
                <div className="flex items-center gap-2 mb-6">
                  <CheckCircle className="h-4 w-4 text-emerald-400" />
                  <p className="text-sm font-bold text-emerald-400">With SHAP Explainability</p>
                </div>
                <div className="space-y-2">
                  {[
                    { label: "Claim filed 3 days after policy",  val: "+33.6%", up: true },
                    { label: "Base policy type — high risk",      val: "+22.6%", up: true },
                    { label: "No police report filed",            val: "+18.1%", up: true },
                    { label: "Driver rating within normal range", val: "−43.1%", up: false },
                    { label: "Claim filed on weekday",            val: "−22.1%", up: false },
                  ].map(({ label, val, up }) => (
                    <div key={label} className={`flex items-center justify-between p-3 rounded-lg border ${up ? "border-red-500/20 bg-red-500/5" : "border-emerald-500/20 bg-emerald-500/5"}`}>
                      <div className="flex items-center gap-2">
                        <div className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${up ? "bg-red-400" : "bg-emerald-400"}`} />
                        <p className="text-xs text-white/70">{label}</p>
                      </div>
                      <span className={`text-xs font-black font-mono ${up ? "text-red-400" : "text-emerald-400"}`}>{val}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* SHAP bar chart */}
            <div className="bg-[#111] border border-white/10 rounded-2xl p-8">
              <p className="text-sm font-bold text-white/60 mb-6">SHAP Impact Visualization — Feature Contribution to Fraud Score</p>
              <div className="space-y-3">
                {[
                  { feature: "Days since policy start",  shap: 0.336, dir: "fraud" },
                  { feature: "Driver rating",             shap: 0.431, dir: "legit" },
                  { feature: "Base policy type",          shap: 0.226, dir: "fraud" },
                  { feature: "Policy number pattern",     shap: 0.452, dir: "legit" },
                  { feature: "No police report",          shap: 0.181, dir: "fraud" },
                  { feature: "Week of month claimed",     shap: 0.318, dir: "legit" },
                  { feature: "Past number of claims",     shap: 0.143, dir: "fraud" },
                  { feature: "Rep number",                shap: 0.271, dir: "legit" },
                ].map(({ feature, shap, dir }) => (
                  <div key={feature} className="flex items-center gap-4">
                    <span className="text-xs text-white/40 w-52 text-right flex-shrink-0 truncate">{feature}</span>
                    <div className="flex-1 flex items-center">
                      {dir === "legit" ? (
                        <div className="flex items-center w-full">
                          <div className="flex-1 h-5 bg-white/5 rounded-r-none rounded-l overflow-hidden flex justify-end">
                            <div className="h-full bg-emerald-500/60 rounded-l" style={{ width: `${shap * 100}%` }} />
                          </div>
                          <div className="w-px h-5 bg-white/20" />
                          <div className="flex-1 h-5 bg-white/5 rounded-l-none rounded-r" />
                        </div>
                      ) : (
                        <div className="flex items-center w-full">
                          <div className="flex-1 h-5 bg-white/5 rounded-r-none rounded-l" />
                          <div className="w-px h-5 bg-white/20" />
                          <div className="flex-1 h-5 bg-white/5 rounded-l-none rounded-r overflow-hidden">
                            <div className="h-full bg-red-500/60 rounded-r" style={{ width: `${shap * 100}%` }} />
                          </div>
                        </div>
                      )}
                    </div>
                    <span className={`text-xs font-mono w-14 flex-shrink-0 ${dir === "fraud" ? "text-red-400" : "text-emerald-400"}`}>
                      {dir === "fraud" ? "+" : "−"}{(shap * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
                <div className="flex items-center gap-4 pt-2 border-t border-white/5">
                  <span className="text-xs text-white/20 w-52 text-right">← Reduces fraud risk</span>
                  <div className="flex-1 flex justify-center"><div className="w-px h-4 bg-white/20" /></div>
                  <span className="text-xs text-white/20 w-14">Increases →</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── FUSION ── */}
      <section className="py-32 px-6 border-t border-white/5">
        <div ref={fusionRef} className="max-w-6xl mx-auto">
          <div className={`transition-all duration-700 ${fusionInView ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8"}`}>
            <div className="text-xs font-bold tracking-widest text-amber-400 uppercase mb-4">The Formula</div>
            <h2 className="text-4xl md:text-6xl font-black tracking-tight mb-6">
              Three models.<br /><span className="text-white/30">One verdict.</span>
            </h2>
            <p className="text-white/50 text-lg mb-16 max-w-2xl">The Fusion Score combines YOLO damage evidence, CNN severity analysis, and XGBoost fraud probability into a single defensible claim decision.</p>

            <div className="grid md:grid-cols-3 gap-6 mb-8">
              {[
                { model: "YOLO", role: "Parts & Damage", output: "₹52,400 base", icon: Camera, color: "blue" },
                { model: "CNN",  role: "Severity Score", output: "1.67× multiplier", icon: Layers, color: "violet" },
                { model: "XGBoost", role: "Fraud Score", output: "73% fraud risk", icon: BarChart3, color: "red" },
              ].map(({ model, role, output, icon: Icon, color }) => (
                <div key={model} className={`bg-[#111] border border-${color}-500/20 rounded-2xl p-6`}>
                  <Icon className={`h-6 w-6 text-${color}-400 mb-4`} />
                  <p className={`text-xs font-bold text-${color}-400 uppercase tracking-wider mb-1`}>{model}</p>
                  <p className="text-white font-semibold mb-3">{role}</p>
                  <p className={`text-lg font-black text-${color}-300`}>{output}</p>
                </div>
              ))}
            </div>

            <div className="flex items-center gap-4 mb-8">
              <div className="flex-1 h-px bg-white/10" />
              <span className="text-white/30 text-sm">combined into</span>
              <div className="flex-1 h-px bg-white/10" />
            </div>

            <div className="bg-gradient-to-r from-amber-500/10 to-amber-600/5 border border-amber-500/20 rounded-2xl p-8 text-center">
              <p className="text-xs font-bold text-amber-400 uppercase tracking-wider mb-2">Final Claim Decision</p>
              <div className="flex items-center justify-center gap-8 flex-wrap">
                <div>
                  <p className="text-5xl font-black text-white">₹87,500</p>
                  <p className="text-xs text-white/40 mt-1">Calculated claim amount</p>
                </div>
                <div className="w-px h-12 bg-white/10" />
                <div>
                  <p className="text-5xl font-black text-red-400">73%</p>
                  <p className="text-xs text-white/40 mt-1">Fraud probability</p>
                </div>
                <div className="w-px h-12 bg-white/10" />
                <div>
                  <p className="text-2xl font-black text-amber-400">Flagged</p>
                  <p className="text-xs text-white/40 mt-1">Admin review required</p>
                </div>
              </div>
              <p className="text-white/30 text-xs mt-6">+ Full SHAP explanation available · PDF report generated · Claim history logged</p>
            </div>
          </div>
        </div>
      </section>

      {/* ── STATS ── */}
      <section id="impact" className="py-32 px-6 border-t border-white/5 bg-[#0d0d0d]">
        <div ref={statsRef} className="max-w-6xl mx-auto">
          <div className="text-xs font-bold tracking-widest text-white/30 uppercase mb-4">Impact</div>
          <h2 className="text-4xl md:text-6xl font-black tracking-tight mb-16">Numbers that matter.</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            {[
              { value: `${claims.toLocaleString()}+`, label: "Claims Processed", sub: "and counting" },
              { value: `${accuracy}%`,               label: "Fraud Detection Accuracy", sub: "vs 60% industry avg" },
              { value: `₹${saved.toLocaleString()}Cr`, label: "Fraud Prevented Annually", sub: "across Indian insurers" },
              { value: `${speed}s`,                  label: "Average Analysis Time", sub: "vs 3 weeks traditional" },
            ].map(({ value, label, sub }) => (
              <div key={label} className="bg-[#111] border border-white/8 rounded-2xl p-6 md:p-8">
                <p className="text-3xl md:text-4xl font-black text-white mb-2">{value}</p>
                <p className="text-white/60 font-semibold text-sm mb-1">{label}</p>
                <p className="text-white/25 text-xs">{sub}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA ── */}
      <section className="py-32 px-6 border-t border-white/5">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-5xl md:text-7xl font-black tracking-tighter mb-6">
            Ready to file<br />
            <span style={{ WebkitTextStroke: "1px rgba(255,255,255,0.3)", color: "transparent" }}>your claim?</span>
          </h2>
          <p className="text-white/40 text-xl mb-10">Join thousands of policyholders who get decisions in hours, not weeks.</p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link href="/register" className="inline-flex items-center gap-2 bg-white text-black font-black px-10 py-5 rounded-xl hover:bg-gray-100 transition-all text-base">
              File a Claim Now <ArrowRight className="h-5 w-5" />
            </Link>
            <Link href="/login" className="inline-flex items-center gap-2 border border-white/20 text-white/60 hover:text-white hover:border-white/40 font-medium px-10 py-5 rounded-xl transition-all text-base">
              Admin Login
            </Link>
          </div>
        </div>
      </section>

      {/* ── FOOTER ── */}
      <footer className="border-t border-white/5 py-12 px-6">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 bg-white rounded-lg flex items-center justify-center">
              <Shield className="h-3.5 w-3.5 text-black" />
            </div>
            <span className="font-bold text-white">SmartClaim AI</span>
          </div>
          <p className="text-white/20 text-xs text-center">
            Transparent · Explainable · Automated · © {new Date().getFullYear()} SmartClaim AI. All rights reserved.
          </p>
          <div className="flex gap-6 text-xs text-white/30">
            <Link href="/login" className="hover:text-white transition-colors">Login</Link>
            <Link href="/register" className="hover:text-white transition-colors">Register</Link>
          </div>
        </div>
      </footer>
    </div>
  );
}

// Fix: add X import used in component
function X({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
    </svg>
  );
}