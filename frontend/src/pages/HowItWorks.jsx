import React from "react";
import {
  Upload,
  MousePointerClick,
  Ruler,
  BarChart2,
  FileText,
  ShieldAlert,
  Cpu,
  Stethoscope,
  ChevronRight,
} from "lucide-react";

const steps = [
  {
    icon: Upload,
    number: "01",
    title: "Upload Angiogram",
    description:
      "Import a coronary angiogram image (JPEG, PNG, or DICOM-derived still frame) directly from your workstation. The image is processed entirely in your browser � nothing is uploaded to a remote server.",
  },
  {
    icon: MousePointerClick,
    number: "02",
    title: "Select Segment Points",
    description:
      "Click to place Point A at the proximal end of the lesion and Point B at the distal end. Use pan and zoom controls to position your markers precisely on the vessel of interest.",
  },
  {
    icon: Ruler,
    number: "03",
    title: "Enter Catheter Size",
    description:
      "Provide the French (Fr) size of the guiding catheter visible in the image. This known reference diameter allows the system to calculate a calibrated pixel-to-millimetre scale.",
  },
  {
    icon: Cpu,
    number: "04",
    title: "Automated Analysis",
    description:
      "The AI engine processes the selected vessel segment � tracing its centreline, computing lumen diameter profiles, and identifying the narrowest cross-section relative to the reference diameter.",
  },
  {
    icon: BarChart2,
    number: "05",
    title: "Review Results",
    description:
      "Examine the quantitative output: lesion length, minimum lumen diameter, and percent diameter stenosis. Use the post-analysis magnifier to inspect fine vessel detail at any location.",
  },
  {
    icon: FileText,
    number: "06",
    title: "Generate Report",
    description:
      "Export a structured summary of the analysis for documentation or review. The report captures all quantitative measurements alongside the annotated segment image.",
  },
];

const principles = [
  {
    icon: Stethoscope,
    title: "Quantitative Coronary Analysis (QCA)",
    body:
      "QCA is the gold-standard method for objectively measuring coronary lesion severity from angiographic images. By comparing lumen dimensions at the lesion site to an interpolated reference diameter, QCA removes observer subjectivity from visual estimates.",
  },
  {
    icon: Ruler,
    title: "Catheter-Based Calibration",
    body:
      "Guiding catheters have known physical diameters. Measuring their pixel width in the image yields a reliable px/mm conversion factor that is applied to all subsequent measurements in the same frame.",
  },
  {
    icon: Cpu,
    title: "AI-Assisted Segmentation",
    body:
      "A deep-learning model trained on annotated coronary angiogram frames performs vessel edge detection. The resulting lumen boundary is used to derive diameter profiles along the selected segment.",
  },
];

export default function HowItWorks({ onNavigate }) {
  return (
    <div className="max-w-5xl mx-auto space-y-16 py-4 pb-16 animate-fade-in">
      {/* Hero */}
      <div className="text-center space-y-4">
        <span className="inline-block bg-teal/10 text-teal text-xs font-semibold px-4 py-1.5 rounded-full tracking-widest uppercase">
          Methodology
        </span>
        <h1 className="text-4xl font-bold text-charcoal-blue leading-tight">
          How Coronix Works
        </h1>
        <p className="text-muted-teal max-w-2xl mx-auto text-base leading-relaxed">
          Coronix is a research-prototype tool for AI-assisted quantitative coronary analysis.
          Below is a step-by-step overview of the complete analysis workflow.
        </p>
      </div>

      {/* Steps */}
      <div className="relative">
        <div className="absolute left-8 top-10 bottom-10 w-px bg-gradient-to-b from-teal/30 via-muted-teal/20 to-transparent hidden md:block" />
        <div className="space-y-6">
          {steps.map((step, idx) => {
            const Icon = step.icon;
            return (
              <div
                key={idx}
                className="relative flex gap-6 bg-white/60 backdrop-blur-sm border border-white/60 rounded-2xl p-6 shadow-soft hover:shadow-md transition-shadow duration-300 group"
              >
                <div className="shrink-0 w-14 h-14 rounded-2xl bg-teal/10 border border-teal/20 flex items-center justify-center group-hover:bg-teal/20 transition-colors">
                  <Icon className="w-6 h-6 text-teal" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-3 mb-1">
                    <span className="text-xs font-bold text-teal/50 tracking-widest">
                      STEP {step.number}
                    </span>
                  </div>
                  <h3 className="text-charcoal-blue font-semibold text-lg mb-1.5">
                    {step.title}
                  </h3>
                  <p className="text-muted-teal text-sm leading-relaxed">
                    {step.description}
                  </p>
                </div>
                <ChevronRight className="w-5 h-5 text-teal/30 shrink-0 self-center group-hover:text-teal/60 transition-colors hidden sm:block" />
              </div>
            );
          })}
        </div>
      </div>

      {/* Underlying Principles */}
      <div className="space-y-6">
        <div className="text-center space-y-2">
          <span className="inline-block bg-rose/10 text-rose text-xs font-semibold px-4 py-1.5 rounded-full tracking-widest uppercase">
            Science
          </span>
          <h2 className="text-2xl font-bold text-charcoal-blue">Underlying Principles</h2>
          <p className="text-muted-teal text-sm">
            The core scientific concepts that power the analysis engine.
          </p>
        </div>
        <div className="grid md:grid-cols-3 gap-5">
          {principles.map((p, idx) => {
            const Icon = p.icon;
            return (
              <div
                key={idx}
                className="bg-white/60 backdrop-blur-sm border border-white/60 rounded-2xl p-6 shadow-soft hover:shadow-md transition-shadow duration-300 space-y-3"
              >
                <div className="w-11 h-11 rounded-xl bg-teal/10 border border-teal/20 flex items-center justify-center">
                  <Icon className="w-5 h-5 text-teal" />
                </div>
                <h3 className="text-charcoal-blue font-semibold">{p.title}</h3>
                <p className="text-muted-teal text-sm leading-relaxed">{p.body}</p>
              </div>
            );
          })}
        </div>
      </div>

      {/* Disclaimer */}
      <div className="bg-rose/5 border border-rose/20 rounded-2xl p-6 flex gap-4">
        <ShieldAlert className="w-6 h-6 text-rose shrink-0 mt-0.5" />
        <div className="space-y-1">
          <p className="text-charcoal-blue font-semibold text-sm">Important Notice</p>
          <p className="text-muted-teal text-sm leading-relaxed">
            This tool is an AI-assisted research/hackathon prototype for demonstration purposes
            only. It is not a medical device, does not provide a diagnosis, and must not be used
            for clinical decision-making.
          </p>
        </div>
      </div>

      {/* CTA */}
      <div className="text-center">
        <button
          onClick={() => onNavigate("analyze")}
          className="inline-flex items-center gap-2 bg-teal text-white font-semibold px-8 py-3.5 rounded-xl hover:bg-teal/90 active:scale-95 transition-all shadow-md"
        >
          Start an Analysis
          <ChevronRight className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
