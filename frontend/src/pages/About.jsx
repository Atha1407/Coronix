import React from "react";
import {
  HeartPulse,
  FlaskConical,
  ShieldAlert,
  Lightbulb,
  Code2,
  ChevronRight,
} from "lucide-react";



const techStack = [
  { label: "Frontend", value: "React 18 + Vite + Tailwind CSS" },
  { label: "AI Engine", value: "Python � OpenCV � PyTorch (backend)" },
  { label: "Report Layer", value: "ReportLab PDF (backend � in progress)" },
  { label: "Deployment Target", value: "Local / on-premise (no cloud PHI)" },
];

const values = [
  {
    icon: FlaskConical,
    title: "Research-First",
    body:
      "Coronix was built for a medical-AI hackathon with a focus on reproducible, explainable methodology over black-box prediction.",
  },
  {
    icon: ShieldAlert,
    title: "Safety by Design",
    body:
      "Every screen carries a prominent disclaimer. The tool is intentionally scoped to quantitative measurement assistance � not diagnostic decision-making.",
  },
  {
    icon: Lightbulb,
    title: "Open Process",
    body:
      "Analysis logic is designed to be inspectable. The codebase is structured so that any clinician or researcher can audit each step of the QCA pipeline.",
  },
];

export default function About({ onNavigate }) {
  return (
    <div className="max-w-5xl mx-auto space-y-16 py-4 pb-16 animate-fade-in">
      {/* Hero */}
      <div className="text-center space-y-4">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-teal/10 border border-teal/20 mx-auto">
          <HeartPulse className="w-8 h-8 text-teal" />
        </div>
        <h1 className="text-4xl font-bold text-charcoal-blue leading-tight">
          About Coronix
        </h1>
        <p className="text-muted-teal max-w-2xl mx-auto text-base leading-relaxed">
          An AI-assisted quantitative coronary analysis prototype, built to explore the
          intersection of deep learning and interventional cardiology workflows.
        </p>
      </div>

      {/* What is Coronix */}
      <div className="bg-white/60 backdrop-blur-sm border border-white/60 rounded-2xl p-8 shadow-soft space-y-4">
        <h2 className="text-xl font-bold text-charcoal-blue">What is Coronix?</h2>
        <p className="text-muted-teal text-sm leading-relaxed">
          Coronix is a browser-based tool that applies computer-vision and AI techniques to
          still-frame coronary angiogram images. Given a user-selected vessel segment and a
          catheter-based calibration reference, it computes quantitative measurements including
          lesion length, minimum lumen diameter, and percent diameter stenosis � the same metrics
          used in clinical Quantitative Coronary Analysis (QCA).
        </p>
        <p className="text-muted-teal text-sm leading-relaxed">
          The project was initiated as a hackathon prototype to demonstrate that a lightweight,
          offline-capable QCA assistant could be assembled from open-source components,
          without requiring an expensive commercial workstation.
        </p>
      </div>

      {/* Core Values */}
      <div className="space-y-6">
        <div className="text-center space-y-2">
          <span className="inline-block bg-teal/10 text-teal text-xs font-semibold px-4 py-1.5 rounded-full tracking-widest uppercase">
            Principles
          </span>
          <h2 className="text-2xl font-bold text-charcoal-blue">Core Values</h2>
        </div>
        <div className="grid md:grid-cols-3 gap-5">
          {values.map((v, idx) => {
            const Icon = v.icon;
            return (
              <div
                key={idx}
                className="bg-white/60 backdrop-blur-sm border border-white/60 rounded-2xl p-6 shadow-soft hover:shadow-md transition-shadow duration-300 space-y-3"
              >
                <div className="w-11 h-11 rounded-xl bg-teal/10 border border-teal/20 flex items-center justify-center">
                  <Icon className="w-5 h-5 text-teal" />
                </div>
                <h3 className="text-charcoal-blue font-semibold">{v.title}</h3>
                <p className="text-muted-teal text-sm leading-relaxed">{v.body}</p>
              </div>
            );
          })}
        </div>
      </div>

      {/* Tech Stack */}
      <div className="bg-white/60 backdrop-blur-sm border border-white/60 rounded-2xl p-8 shadow-soft space-y-5">
        <div className="flex items-center gap-3">
          <Code2 className="w-5 h-5 text-teal" />
          <h2 className="text-xl font-bold text-charcoal-blue">Technology Stack</h2>
        </div>
        <div className="grid sm:grid-cols-2 gap-4">
          {techStack.map((t, idx) => (
            <div
              key={idx}
              className="flex gap-4 bg-bright-snow/60 rounded-xl px-5 py-4 border border-white/60"
            >
              <div className="flex-1 min-w-0">
                <p className="text-xs text-muted-teal font-semibold uppercase tracking-wider mb-0.5">
                  {t.label}
                </p>
                <p className="text-charcoal-blue text-sm font-medium">{t.value}</p>
              </div>
            </div>
          ))}
        </div>
      </div>



      {/* Disclaimer */}
      <div className="bg-rose/5 border border-rose/20 rounded-2xl p-6 flex gap-4">
        <ShieldAlert className="w-6 h-6 text-rose shrink-0 mt-0.5" />
        <div className="space-y-1">
          <p className="text-charcoal-blue font-semibold text-sm">Medical Disclaimer</p>
          <p className="text-muted-teal text-sm leading-relaxed">
            This tool is an AI-assisted research/hackathon prototype for demonstration purposes
            only. It is not a medical device, does not provide a diagnosis, and must not be used
            for clinical decision-making.
          </p>
        </div>
      </div>

      {/* CTA */}
      <div className="text-center space-y-3">
        <p className="text-muted-teal text-sm">Ready to try it out?</p>
        <button
          onClick={() => onNavigate("analyze")}
          className="inline-flex items-center gap-2 bg-teal text-white font-semibold px-8 py-3.5 rounded-xl hover:bg-teal/90 active:scale-95 transition-all shadow-md"
        >
          Open Analyze
          <ChevronRight className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
