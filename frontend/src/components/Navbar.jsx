import React from 'react';
import { HeartPulse, ChevronDown } from 'lucide-react';

export default function Navbar() {
  return (
    <nav className="h-16 border-b border-white/50 bg-white/40 backdrop-blur-md flex items-center justify-between px-6 shrink-0 z-10">
      <div className="flex items-center gap-8">
        <div className="flex items-center gap-2 text-charcoal-blue">
          <HeartPulse className="w-6 h-6 text-teal" />
          <span className="font-semibold text-lg tracking-tight">CORONIX</span>
        </div>
        
        <div className="hidden md:flex gap-6 text-sm font-medium">
          <button className="text-teal border-b-2 border-teal py-5">Analyze</button>
          <button className="text-muted-teal hover:text-charcoal-blue transition-colors py-5">How It Works</button>
          <button className="text-muted-teal hover:text-charcoal-blue transition-colors py-5">About</button>
        </div>
      </div>

      <div className="flex items-center gap-3">
        <div className="flex flex-col items-end">
          <span className="text-sm font-medium text-charcoal-blue leading-tight">Dr. Sharma</span>
          <span className="text-xs text-muted-teal leading-tight">Cardiologist</span>
        </div>
        <div className="w-9 h-9 rounded-full bg-teal text-white flex items-center justify-center font-medium">
          DS
        </div>
        <ChevronDown className="w-4 h-4 text-muted-teal" />
      </div>
    </nav>
  );
}
