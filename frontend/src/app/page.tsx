"use client";

import { useEffect, useState, useRef } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import {
  Button,
} from "@/components/ui/button";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Separator } from "@/components/ui/separator";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import {
  Camera,
  Brain,
  CheckCircle2,
  Shield,
  FileSearch,
  Users,
  Eye,
  ArrowRight,
  Menu,
  BarChart3,
  Sparkles,
  Lock,
  Building2,
} from "lucide-react";

// Animated counter hook
function useCountUp(end: number, duration: number, trigger: boolean) {
  const [count, setCount] = useState(0);
  useEffect(() => {
    if (!trigger) return;
    let start = 0;
    const increment = end / (duration / 16);
    const timer = setInterval(() => {
      start += increment;
      if (start >= end) {
        setCount(end);
        clearInterval(timer);
      } else {
        setCount(Math.floor(start));
      }
    }, 16);
    return () => clearInterval(timer);
  }, [end, duration, trigger]);
  return count;
}

export default function HomePage() {
  const router = useRouter();
  const statsRef = useRef<HTMLDivElement>(null);
  const [statsInView, setStatsInView] = useState(false);

  useEffect(() => {
    const token = localStorage.getItem("access_token");
    const role = localStorage.getItem("user_role")?.toLowerCase();
    if (token) {
      if (role === "admin") router.push("/admin");
      else if (role === "surveyor") router.push("/surveyor");
      else router.push("/customer");
      return;
    }
  }, [router]);

  useEffect(() => {
    const obs = new IntersectionObserver(
      ([e]) => setStatsInView(e.isIntersecting),
      { threshold: 0.3 }
    );
    if (statsRef.current) obs.observe(statsRef.current);
    return () => obs.disconnect();
  }, []);

  const claimsProcessed = useCountUp(10000, 1500, statsInView);
  const avgTime = useCountUp(48, 1200, statsInView);
  const fraudAccuracy = useCountUp(96, 1500, statsInView);
  const policyholders = useCountUp(50000, 1800, statsInView);

  const navLinks = [
    { text: "How it Works", href: "#how-it-works" },
    { text: "Features", href: "#features" },
    { text: "Security", href: "#security" },
    { text: "FAQ", href: "#faq" },
  ];

  const sectionWrapper = "w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8";

  return (
    <div className="bg-gray-50 text-gray-900 min-h-screen w-full antialiased">
      {/* Navbar */}
      <header className="sticky top-0 z-50 border-b border-gray-200 bg-white/95 backdrop-blur-md shadow-sm">
        <div className={`${sectionWrapper} flex h-16 items-center justify-between`}>
          <Link href="/" className="flex items-center gap-2.5 font-bold text-gray-900 hover:text-gray-900 transition-colors">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-black text-white">
              <Shield className="h-5 w-5" />
            </div>
            SmartClaim
          </Link>
          <nav className="hidden md:flex items-center gap-8">
            {navLinks.map((link) => (
              <a
                key={link.href}
                href={link.href}
                className="text-sm font-medium text-gray-600 hover:text-gray-900 transition-colors"
              >
                {link.text}
              </a>
            ))}
          </nav>
          <div className="flex items-center gap-3">
            <Button asChild variant="ghost" size="sm" className="text-gray-600 hover:text-gray-900">
              <Link href="/login">Login</Link>
            </Button>
            <Button asChild size="sm" className="bg-black hover:bg-gray-800 text-white shadow-md">
              <Link href="/register">Get Started</Link>
            </Button>
            <Sheet>
              <SheetTrigger asChild className="md:hidden">
                <Button variant="ghost" size="icon">
                  <Menu className="h-5 w-5" />
                </Button>
              </SheetTrigger>
              <SheetContent side="right" className="w-[280px]">
                <nav className="flex flex-col gap-4 mt-8">
                  {navLinks.map((link) => (
                    <a key={link.href} href={link.href} className="text-lg">
                      {link.text}
                    </a>
                  ))}
                  <Separator />
                  <Button asChild variant="ghost" className="justify-start">
                    <Link href="/login">Login</Link>
                  </Button>
                  <Button asChild className="justify-start">
                    <Link href="/register">Get Started</Link>
                  </Button>
                </nav>
              </SheetContent>
            </Sheet>
          </div>
        </div>
      </header>

      {/* Hero */}
      <section className="relative bg-black text-white py-28 lg:py-36 overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_80%_at_50%_-20%,rgba(255,255,255,0.05),transparent_50%)]" />
        <div className="absolute inset-0 bg-[url('data:image/svg+xml,%3Csvg width=\'60\' height=\'60\' viewBox=\'0 0 60 60\' xmlns=\'http://www.w3.org/2000/svg\'%3E%3Cg fill=\'none\' fill-rule=\'evenodd\'%3E%3Cg fill=\'%23ffffff\' fill-opacity=\'0.03\'%3E%3Cpath d=\'M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z\'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E')] opacity-40" />
        <div className={`${sectionWrapper} relative`}>
          <div className="max-w-3xl">
            <Badge className="mb-6 bg-white/20 text-gray-300 border-white/20">
              AI-Powered • Settled in Hours
            </Badge>
            <h1 className="text-4xl md:text-6xl lg:text-7xl font-extrabold tracking-tight mb-6 leading-[1.1]">
              Insurance Claims.
              <br />
              <span className="bg-gradient-to-r from-gray-300 to-gray-400 bg-clip-text text-transparent">Settled in Hours.</span>
            </h1>
            <p className="text-lg md:text-xl text-gray-300 mb-6 max-w-xl leading-relaxed">
              Submit photos, get an instant AI assessment, and track your claim through every step. No paperwork, no waiting.
            </p>
            <p className="text-gray-400 mb-10 max-w-xl">
              Trusted by leading insurers. Built with computer vision and ML for accurate damage assessment and fraud detection.
            </p>
            <div className="flex flex-col sm:flex-row gap-4">
              <Button asChild size="lg" className="bg-white text-gray-900 hover:bg-gray-100 shadow-xl shadow-black/20 font-semibold h-12 px-8 rounded-xl">
                <Link href="/register">File a Claim</Link>
              </Button>
              <Button asChild size="lg" variant="outline" className="border-gray-500/50 text-gray-200 hover:bg-white/10 hover:border-gray-400 h-12 px-8 rounded-xl font-medium">
                <Link href="/login">Admin Login</Link>
              </Button>
            </div>
          </div>
          {/* Hero visual - mock claim card */}
          <div className="mt-16 lg:mt-0 lg:absolute lg:right-0 lg:top-1/2 lg:-translate-y-1/2 lg:w-[380px]">
            <Card className="w-full bg-white/5 border border-white/10 backdrop-blur-sm shadow-2xl overflow-hidden">
              <div className="absolute top-0 right-0 w-32 h-32 bg-white/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2" />
              <CardHeader className="pb-2 relative">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm font-medium text-gray-300">Claim Status</CardTitle>
                  <Badge className="bg-white/20 text-gray-300 border-white/30">Verified</Badge>
                </div>
              </CardHeader>
              <CardContent className="space-y-5 relative">
                <div>
                  <p className="text-3xl font-bold text-white">₹45,000</p>
                  <p className="text-xs text-gray-400">Estimated amount</p>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-xs text-gray-400">
                    <span>Confidence</span>
                    <span>72%</span>
                  </div>
                  <div className="h-2.5 bg-gray-700/50 rounded-full overflow-hidden">
                    <div className="h-full w-[72%] bg-gradient-to-r from-gray-400 to-gray-500 rounded-full transition-all duration-500" />
                  </div>
                </div>
                <p className="text-xs text-gray-500">CLM-2024-001234 • Honda City</p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* How it Works */}
      <section id="how-it-works" className="py-24 lg:py-32 bg-white">
        <div className={sectionWrapper}>
          <div className="text-center mb-20">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">How It Works</h2>
            <p className="text-gray-600 max-w-2xl mx-auto text-lg">
              Three simple steps from submission to decision. No complexity, no guesswork.
            </p>
          </div>
          <div className="grid md:grid-cols-3 gap-8">
            <Card className="border-0 shadow-lg shadow-gray-200/50 rounded-2xl overflow-hidden hover:shadow-xl hover:shadow-gray-200/50 transition-all duration-300 group">
              <CardHeader className="text-center pb-6 pt-8">
                <div className="mx-auto mb-4 w-16 h-16 rounded-2xl bg-gray-200 flex items-center justify-center group-hover:bg-black transition-colors">
                  <Camera className="h-8 w-8 text-gray-900 group-hover:text-white transition-colors" />
                </div>
                <CardTitle className="text-xl text-gray-900">1. Submit Photos</CardTitle>
                <CardDescription className="text-gray-600 mt-2 leading-relaxed">
                  Upload damage photos directly from your phone or computer. Our AI analyzes vehicle damage, detects affected parts, and validates image authenticity in seconds.
                </CardDescription>
              </CardHeader>
            </Card>
            <Card className="border-0 shadow-lg shadow-gray-200/50 rounded-2xl overflow-hidden hover:shadow-xl hover:shadow-gray-200/50 transition-all duration-300 group">
              <CardHeader className="text-center pb-6 pt-8">
                <div className="mx-auto mb-4 w-16 h-16 rounded-2xl bg-gray-200 flex items-center justify-center group-hover:bg-black transition-colors">
                  <Brain className="h-8 w-8 text-gray-900 group-hover:text-white transition-colors" />
                </div>
                <CardTitle className="text-xl text-gray-900">2. AI Analysis</CardTitle>
                <CardDescription className="text-gray-600 mt-2 leading-relaxed">
                  YOLO object detection identifies damage regions. CNN models assess severity. Our XGBoost fraud model scores risk in real time.
                </CardDescription>
              </CardHeader>
            </Card>
            <Card className="border-0 shadow-lg shadow-gray-200/50 rounded-2xl overflow-hidden hover:shadow-xl hover:shadow-gray-200/50 transition-all duration-300 group">
              <CardHeader className="text-center pb-6 pt-8">
                <div className="mx-auto mb-4 w-16 h-16 rounded-2xl bg-gray-200 flex items-center justify-center group-hover:bg-black transition-colors">
                  <CheckCircle2 className="h-8 w-8 text-gray-900 group-hover:text-white transition-colors" />
                </div>
                <CardTitle className="text-xl text-gray-900">3. Decision</CardTitle>
                <CardDescription className="text-gray-600 mt-2 leading-relaxed">
                  Receive verification, get flagged for manual review, or be routed to a field surveyor. Track every step in your portal.
                </CardDescription>
              </CardHeader>
            </Card>
          </div>
        </div>
      </section>

      {/* Stats */}
      <section ref={statsRef} className="py-20 lg:py-24 bg-gray-100">
        <div className={sectionWrapper}>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-6 lg:gap-8">
            <div className="text-center p-6 rounded-2xl bg-white shadow-md border border-gray-100">
              <p className="text-3xl lg:text-4xl font-bold text-gray-900">{claimsProcessed.toLocaleString()}+</p>
              <p className="text-sm text-gray-600 mt-1 font-medium">Claims processed</p>
            </div>
            <div className="text-center p-6 rounded-2xl bg-white shadow-md border border-gray-100">
              <p className="text-3xl lg:text-4xl font-bold text-gray-900">{avgTime}h</p>
              <p className="text-sm text-gray-600 mt-1 font-medium">Avg. settlement time</p>
            </div>
            <div className="text-center p-6 rounded-2xl bg-white shadow-md border border-gray-100">
              <p className="text-3xl lg:text-4xl font-bold text-gray-900">{fraudAccuracy}%</p>
              <p className="text-sm text-gray-600 mt-1 font-medium">Fraud detection accuracy</p>
            </div>
            <div className="text-center p-6 rounded-2xl bg-white shadow-md border border-gray-100">
              <p className="text-3xl lg:text-4xl font-bold text-gray-900">{policyholders.toLocaleString()}+</p>
              <p className="text-sm text-gray-600 mt-1 font-medium">Policyholders protected</p>
            </div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section id="features" className="py-24 lg:py-32 bg-gray-50">
        <div className={sectionWrapper}>
          <div className="text-center mb-20">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">Why SmartClaim</h2>
            <p className="text-gray-600 max-w-2xl mx-auto text-lg">
              Built for speed, transparency, and trust. Every decision is traceable.
            </p>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            <Card className="border-0 bg-white shadow-md hover:shadow-lg transition-shadow rounded-xl overflow-hidden group">
              <CardHeader className="group-hover:bg-gray-50 transition-colors">
                <div className="w-12 h-12 rounded-xl bg-gray-200 flex items-center justify-center mb-3">
                  <FileSearch className="h-6 w-6 text-gray-900" />
                </div>
                <CardTitle className="text-lg text-gray-900">YOLO + CNN Damage Detection</CardTitle>
                <CardDescription className="text-gray-600">
                  AI reads your photos. Damage regions detected and assessed automatically—no manual estimate delays.
                </CardDescription>
              </CardHeader>
            </Card>
            <Card className="border-0 bg-white shadow-md hover:shadow-lg transition-shadow rounded-xl overflow-hidden group">
              <CardHeader className="group-hover:bg-gray-50 transition-colors">
                <div className="w-12 h-12 rounded-xl bg-gray-200 flex items-center justify-center mb-3">
                  <BarChart3 className="h-6 w-6 text-gray-900" />
                </div>
                <CardTitle className="text-lg text-gray-900">Real-time Fraud Scoring</CardTitle>
                <CardDescription className="text-gray-600">
                  XGBoost model flags suspicious patterns. Risk levels and confidence scores shown transparently.
                </CardDescription>
              </CardHeader>
            </Card>
            <Card className="border-0 bg-white shadow-md hover:shadow-lg transition-shadow rounded-xl overflow-hidden group">
              <CardHeader className="group-hover:bg-gray-50 transition-colors">
                <div className="w-12 h-12 rounded-xl bg-gray-200 flex items-center justify-center mb-3">
                  <Users className="h-6 w-6 text-gray-900" />
                </div>
                <CardTitle className="text-lg text-gray-900">Field Surveyor Network</CardTitle>
                <CardDescription className="text-gray-600">
                  Physical verification when needed. Seamless handoff from AI to human for complex cases.
                </CardDescription>
              </CardHeader>
            </Card>
            <Card className="border-0 bg-white shadow-md hover:shadow-lg transition-shadow rounded-xl overflow-hidden group">
              <CardHeader className="group-hover:bg-gray-50 transition-colors">
                <div className="w-12 h-12 rounded-xl bg-gray-200 flex items-center justify-center mb-3">
                  <Eye className="h-6 w-6 text-gray-900" />
                </div>
                <CardTitle className="text-lg text-gray-900">Transparent Process</CardTitle>
                <CardDescription className="text-gray-600">
                  Track every step in your portal. Notifications, timeline, and PDF reports—never left guessing.
                </CardDescription>
              </CardHeader>
            </Card>
          </div>
        </div>
      </section>

      {/* Security & Trust */}
      <section id="security" className="py-24 lg:py-32 bg-white">
        <div className={sectionWrapper}>
          <div className="text-center mb-20">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">Security & Compliance</h2>
            <p className="text-gray-600 max-w-2xl mx-auto text-lg">
              Your data is protected with enterprise-grade security. We follow industry standards for insurance and financial services.
            </p>
          </div>
          <div className="grid md:grid-cols-3 gap-8">
            <Card className="border-0 shadow-lg rounded-2xl overflow-hidden text-center hover:shadow-xl transition-shadow">
              <CardHeader className="pt-10 pb-8">
                <div className="mx-auto w-14 h-14 rounded-2xl bg-gray-200 flex items-center justify-center mb-4">
                  <Lock className="h-7 w-7 text-gray-700" />
                </div>
                <CardTitle className="text-lg text-gray-900">Encrypted & Secure</CardTitle>
                <CardDescription className="text-gray-600 mt-2">
                  All data encrypted in transit and at rest. Photos and documents stored securely with strict access controls.
                </CardDescription>
              </CardHeader>
            </Card>
            <Card className="border-0 shadow-lg rounded-2xl overflow-hidden text-center hover:shadow-xl transition-shadow">
              <CardHeader className="pt-10 pb-8">
                <div className="mx-auto w-14 h-14 rounded-2xl bg-gray-200 flex items-center justify-center mb-4">
                  <Shield className="h-7 w-7 text-gray-700" />
                </div>
                <CardTitle className="text-lg text-gray-900">Fraud Prevention</CardTitle>
                <CardDescription className="text-gray-600 mt-2">
                  Advanced ML models detect tampering, duplicate claims, and suspicious patterns.
                </CardDescription>
              </CardHeader>
            </Card>
            <Card className="border-0 shadow-lg rounded-2xl overflow-hidden text-center hover:shadow-xl transition-shadow">
              <CardHeader className="pt-10 pb-8">
                <div className="mx-auto w-14 h-14 rounded-2xl bg-gray-200 flex items-center justify-center mb-4">
                  <Building2 className="h-7 w-7 text-gray-700" />
                </div>
                <CardTitle className="text-lg text-gray-900">Insurer-Ready</CardTitle>
                <CardDescription className="text-gray-600 mt-2">
                  Built for integration with existing policy systems. APIs, audit logs, and compliance reporting included.
                </CardDescription>
              </CardHeader>
            </Card>
          </div>
        </div>
      </section>

      {/* How the AI Works */}
      <section className="py-24 lg:py-32 bg-gray-50">
        <div className={sectionWrapper}>
          <div className="max-w-3xl mx-auto text-center">
            <Badge className="mb-6 bg-gray-200 text-gray-800 border-0">How the AI Works</Badge>
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-6">Your Photo → Fair Amount</h2>
            <p className="text-gray-600 mb-12 text-lg">
              The amount isn’t arbitrary. AI detects parts, assesses damage, and calculates based on repair costs.
            </p>
            <div className="flex flex-wrap justify-center gap-3 sm:gap-6 text-sm">
              <div className="flex items-center gap-2 bg-white shadow-md border border-gray-100 px-5 py-3 rounded-xl">
                <span className="font-medium text-gray-700">Your photo</span>
                <ArrowRight className="h-4 w-4 text-gray-600 shrink-0" />
              </div>
              <div className="flex items-center gap-2 bg-white shadow-md border border-gray-100 px-5 py-3 rounded-xl">
                <span className="font-medium text-gray-700">Parts detected</span>
                <ArrowRight className="h-4 w-4 text-gray-600 shrink-0" />
              </div>
              <div className="flex items-center gap-2 bg-white shadow-md border border-gray-100 px-5 py-3 rounded-xl">
                <span className="font-medium text-gray-700">Damage assessed</span>
                <ArrowRight className="h-4 w-4 text-gray-600 shrink-0" />
              </div>
              <div className="flex items-center gap-2 bg-white shadow-md border border-gray-100 px-5 py-3 rounded-xl">
                <span className="font-medium text-gray-700">Amount calculated</span>
              </div>
            </div>
            <p className="text-gray-600 text-sm mt-10 max-w-xl mx-auto leading-relaxed">
              Our models use millions of labeled images and repair cost data. Results include confidence scores and a clear breakdown you can download as a PDF report.
            </p>
          </div>
        </div>
      </section>

      {/* Testimonials */}
      <section className="py-24 lg:py-32 bg-white">
        <div className={sectionWrapper}>
          <div className="text-center mb-20">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">What Policyholders Say</h2>
            <p className="text-gray-600 max-w-2xl mx-auto text-lg">
              Real feedback from users who’ve experienced the SmartClaim difference.
            </p>
          </div>
          <div className="grid md:grid-cols-3 gap-8">
            <Card className="border border-gray-100 shadow-lg rounded-2xl overflow-hidden">
              <CardHeader className="relative">
                <div className="absolute top-6 right-6 flex gap-0.5">
                  {[...Array(5)].map((_, i) => (
                    <Sparkles key={i} className="h-4 w-4 fill-gray-400 text-gray-400" />
                  ))}
                </div>
                <CardDescription className="text-gray-600 pr-12 leading-relaxed">
                  "Filed my claim at 9 AM, had a decision by noon. The AI analysis was surprisingly accurate. Best experience with insurance ever."
                </CardDescription>
                <div className="flex items-center gap-3 pt-6 mt-4 border-t border-gray-100">
                  <Avatar className="h-11 w-11 rounded-full bg-gray-200 text-gray-700 font-semibold">
                    <AvatarFallback>RK</AvatarFallback>
                  </Avatar>
                  <div>
                    <p className="font-semibold text-gray-900">Rahul K.</p>
                    <p className="text-xs text-gray-500">Mumbai</p>
                  </div>
                </div>
              </CardHeader>
            </Card>
            <Card className="border border-gray-100 shadow-lg rounded-2xl overflow-hidden">
              <CardHeader className="relative">
                <div className="absolute top-6 right-6 flex gap-0.5">
                  {[...Array(5)].map((_, i) => (
                    <Sparkles key={i} className="h-4 w-4 fill-gray-400 text-gray-400" />
                  ))}
                </div>
                <CardDescription className="text-gray-600 pr-12 leading-relaxed">
                  "No more back-and-forth with adjusters. The portal showed exactly how the amount was calculated. Transparent and fast."
                </CardDescription>
                <div className="flex items-center gap-3 pt-6 mt-4 border-t border-gray-100">
                  <Avatar className="h-11 w-11 rounded-full bg-gray-200 text-gray-700 font-semibold">
                    <AvatarFallback>PS</AvatarFallback>
                  </Avatar>
                  <div>
                    <p className="font-semibold text-gray-900">Priya S.</p>
                    <p className="text-xs text-gray-500">Bangalore</p>
                  </div>
                </div>
              </CardHeader>
            </Card>
            <Card className="border border-gray-100 shadow-lg rounded-2xl overflow-hidden">
              <CardHeader className="relative">
                <div className="absolute top-6 right-6 flex gap-0.5">
                  {[...Array(5)].map((_, i) => (
                    <Sparkles key={i} className="h-4 w-4 fill-gray-400 text-gray-400" />
                  ))}
                </div>
                <CardDescription className="text-gray-600 pr-12 leading-relaxed">
                  "Uploaded 4 photos from my phone. Got a detailed breakdown and PDF report. The field surveyor came within 2 days when needed."
                </CardDescription>
                <div className="flex items-center gap-3 pt-6 mt-4 border-t border-gray-100">
                  <Avatar className="h-11 w-11 rounded-full bg-gray-200 text-gray-700 font-semibold">
                    <AvatarFallback>AM</AvatarFallback>
                  </Avatar>
                  <div>
                    <p className="font-semibold text-gray-900">Amit M.</p>
                    <p className="text-xs text-gray-500">Delhi</p>
                  </div>
                </div>
              </CardHeader>
            </Card>
          </div>
        </div>
      </section>

      {/* FAQ */}
      <section id="faq" className="py-24 lg:py-32 bg-gray-50">
        <div className={sectionWrapper}>
          <div className="max-w-3xl mx-auto">
            <div className="text-center mb-16">
              <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">Frequently Asked Questions</h2>
              <p className="text-gray-600">Common questions about the claim process.</p>
            </div>
            <Accordion type="single" collapsible defaultValue="q1" className="w-full rounded-2xl border border-gray-200 bg-white shadow-lg overflow-hidden">
              <AccordionItem value="q1" className="border-b px-4 last:border-b-0">
                <AccordionTrigger>How long does claim processing take?</AccordionTrigger>
                <AccordionContent>
                  Most straight-forward claims receive an AI assessment within minutes. Verified claims can be settled in 24–48 hours. Cases requiring field survey may take 3–5 business days.
                </AccordionContent>
              </AccordionItem>
              <AccordionItem value="q2" className="border-b px-4 last:border-b-0">
                <AccordionTrigger>What documents do I need?</AccordionTrigger>
                <AccordionContent>
                  You need clear photos of the damage from multiple angles. For submission, we also collect vehicle registration, driving license number, and a brief description. No physical paperwork is required.
                </AccordionContent>
              </AccordionItem>
              <AccordionItem value="q3" className="border-b px-4 last:border-b-0">
                <AccordionTrigger>How is the claim amount calculated?</AccordionTrigger>
                <AccordionContent>
                  Our AI detects damaged parts, assesses severity, and estimates repair costs using industry benchmarks. The model factors in vehicle make, model, and damage extent. You can see a breakdown in your claim portal.
                </AccordionContent>
              </AccordionItem>
              <AccordionItem value="q4" className="border-b px-4 last:border-b-0">
                <AccordionTrigger>What happens if I disagree with the decision?</AccordionTrigger>
                <AccordionContent>
                  You can request a review. A human adjuster will re-examine your claim, and we may schedule a field survey for verification. Contact support through the portal or email for escalation.
                </AccordionContent>
              </AccordionItem>
              <AccordionItem value="q5" className="border-b px-4 last:border-b-0">
                <AccordionTrigger>Is my data safe? How do you handle privacy?</AccordionTrigger>
                <AccordionContent>
                  Yes. We use encryption, secure storage, and strict access controls. Photos and personal data are only used for claim processing. We comply with data protection regulations and do not share your information with third parties for marketing.
                </AccordionContent>
              </AccordionItem>
              <AccordionItem value="q6" className="border-b-0 px-4">
                <AccordionTrigger>Can I track my claim in real time?</AccordionTrigger>
                <AccordionContent>
                  Absolutely. Your portal shows a timeline of every status change, from submission to final decision. You’ll receive notifications when your claim moves to a new stage, when a surveyor is assigned, or when a report is ready to download.
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          </div>
        </div>
      </section>

      {/* CTA Banner */}
      <section className="py-24 lg:py-32 bg-black text-white relative overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_50%_at_50%_100%,rgba(255,255,255,0.1),transparent)]" />
        <div className={`${sectionWrapper} text-center relative`}>
          <h2 className="text-3xl md:text-4xl font-bold mb-4">Ready to file your claim?</h2>
          <p className="text-gray-400 mb-10 max-w-xl mx-auto text-lg">
            Join thousands of policyholders who have streamlined their claim experience.
          </p>
          <Button asChild size="lg" className="bg-white text-gray-900 hover:bg-gray-100 font-semibold h-14 px-10 rounded-xl shadow-xl">
            <Link href="/register">Sign Up Now</Link>
          </Button>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-gray-200 bg-gray-900 text-gray-300 py-16">
        <div className={sectionWrapper}>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-12">
            <div className="lg:col-span-2">
              <div className="flex items-center gap-2.5 font-bold text-white mb-4">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gray-700 text-white">
                  <Shield className="h-5 w-5" />
                </div>
                SmartClaim
              </div>
              <p className="text-sm text-gray-400 max-w-md mb-6 leading-relaxed">
                AI-powered insurance claims platform. Submit photos, get instant AI assessment, and track your claim from submission to settlement. Built for insurers and policyholders.
              </p>
              <Badge className="bg-gray-700 text-gray-300 border-0">Powered by AI fraud detection</Badge>
            </div>
            <div>
              <p className="font-semibold text-white text-sm mb-4">Product</p>
              <div className="flex flex-col gap-3">
                <Link href="#how-it-works" className="text-sm text-gray-400 hover:text-white transition-colors">How it Works</Link>
                <Link href="#features" className="text-sm text-gray-400 hover:text-white transition-colors">Features</Link>
                <Link href="#security" className="text-sm text-gray-400 hover:text-white transition-colors">Security</Link>
                <Link href="#faq" className="text-sm text-gray-400 hover:text-white transition-colors">FAQ</Link>
              </div>
            </div>
            <div>
              <p className="font-semibold text-white text-sm mb-4">Account</p>
              <div className="flex flex-col gap-3">
                <Link href="/login" className="text-sm text-gray-400 hover:text-white transition-colors">Login</Link>
                <Link href="/register" className="text-sm text-gray-400 hover:text-white transition-colors">Register</Link>
              </div>
              <p className="font-semibold text-white text-sm mt-8 mb-3">Contact</p>
              <p className="text-sm text-gray-400">support@smartclaim.com</p>
              <p className="text-sm text-gray-400">+91 1800-XXX-XXXX</p>
            </div>
          </div>
          <Separator className="my-12 border-gray-700" />
          <p className="text-center text-sm text-gray-500">© {new Date().getFullYear()} SmartClaim. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}
