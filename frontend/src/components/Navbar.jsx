import React from 'react';
import { HeartPulse, ChevronDown } from 'lucide-react';

const NAV_ITEMS = [
  { id: 'analyze', label: 'Analyze' },
  { id: 'how-it-works', label: 'How It Works' },
  { id: 'about', label: 'About' },
];

export default function Navbar({ activePage = 'analyze', onNavigate }) {
  return (
    <nav className="h-16 border-b border-white/50 bg-white/40 backdrop-blur-md flex items-center justify-between px-6 shrink-0 z-10">
      <div className="flex items-center gap-8">
        {/* Logo — clicking always goes to Analyze */}
        <button
          onClick={() => onNavigate?.('analyze')}
          className="flex items-center gap-2 text-charcoal-blue hover:opacity-80 transition-opacity"
        >
          <HeartPulse className="w-6 h-6 text-teal" />
          <span className="font-semibold text-lg tracking-tight">CORONIX</span>
        </button>

        <div className="hidden md:flex gap-6 text-sm font-medium">
          {NAV_ITEMS.map((item) => {
            const isActive = activePage === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onNavigate?.(item.id)}
                className={`py-5 transition-colors ${
                  isActive
                    ? 'text-teal border-b-2 border-teal'
                    : 'text-muted-teal hover:text-charcoal-blue'
                }`}
              >
                {item.label}
              </button>
            );
          })}
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
