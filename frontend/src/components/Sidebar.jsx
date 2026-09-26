import React from 'react';
import { PlusCircle, FileText, BarChart2, Settings, HelpCircle } from 'lucide-react';

export default function Sidebar() {
  return (
    <aside className="w-64 border-r border-white/50 bg-white/40 backdrop-blur-md flex flex-col shrink-0 z-10">
      <div className="p-6 flex-1">
        <div className="space-y-2">
          <button className="w-full flex items-center gap-3 bg-teal/10 text-teal font-medium px-4 py-3 rounded-xl transition-colors">
            <PlusCircle className="w-5 h-5" />
            New Analysis
          </button>
          <button className="w-full flex items-center gap-3 text-muted-teal hover:bg-white/50 hover:text-charcoal-blue font-medium px-4 py-3 rounded-xl transition-colors">
            <HelpCircle className="w-5 h-5" />
            How It Works
          </button>
          <button className="w-full flex items-center gap-3 text-muted-teal hover:bg-white/50 hover:text-charcoal-blue font-medium px-4 py-3 rounded-xl transition-colors">
            <FileText className="w-5 h-5" />
            About
          </button>
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
