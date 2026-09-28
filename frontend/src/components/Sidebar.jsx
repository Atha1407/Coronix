import React from 'react';
import { PlusCircle, FileText, HelpCircle } from 'lucide-react';

const SIDEBAR_ITEMS = [
  { id: 'analyze', icon: PlusCircle, label: 'New Analysis' },
  { id: 'how-it-works', icon: HelpCircle, label: 'How It Works' },
  { id: 'about', icon: FileText, label: 'About' },
];

export default function Sidebar({ activePage = 'analyze', onNavigate }) {
  return (
    <aside className="w-64 border-r border-white/50 bg-white/40 backdrop-blur-md flex flex-col shrink-0 z-10">
      <div className="p-6 flex-1">
        <div className="space-y-2">
          {SIDEBAR_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive = activePage === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onNavigate?.(item.id)}
                className={`w-full flex items-center gap-3 font-medium px-4 py-3 rounded-xl transition-colors ${
                  isActive
                    ? 'bg-teal/10 text-teal'
                    : 'text-muted-teal hover:bg-white/50 hover:text-charcoal-blue'
                }`}
              >
                <Icon className="w-5 h-5 shrink-0" />
                {item.label}
              </button>
            );
          })}
        </div>
      </div>

      <div className="p-6 mt-auto">
        <div className="bg-white/60 p-4 rounded-xl border border-white/50">
          <div className="flex items-center gap-2 text-charcoal-blue font-medium mb-1">
            <HelpCircle className="w-4 h-4 text-teal" />
            Need Help?
          </div>
          <p className="text-xs text-muted-teal leading-relaxed">
            View tutorial or guidelines for analysis.
          </p>
        </div>
      </div>
    </aside>
  );
}
