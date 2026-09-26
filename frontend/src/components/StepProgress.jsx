import React from 'react';
import { Check } from 'lucide-react';

export default function StepProgress({ currentStep }) {
  const steps = [
    { num: 1, label: 'Upload Angiogram' },
    { num: 2, label: 'Select Two Points' },
    { num: 3, label: 'Analyze & View Results' }
  ];

  return (
    <div className="flex items-center gap-4 mb-8">
      {steps.map((step, index) => {
        const isCompleted = currentStep > step.num;
        const isCurrent = currentStep === step.num;
        const isFuture = currentStep < step.num;

        return (
          <React.Fragment key={step.num}>
            <div className="flex items-center gap-3">
              <div
                className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium transition-colors ${
                  isCompleted
                    ? 'bg-teal text-white'
                    : isCurrent
                    ? 'bg-teal text-white ring-4 ring-teal/20'
                    : 'bg-white text-muted-teal border border-muted-teal/30'
                }`}
              >
                {isCompleted ? <Check className="w-4 h-4" /> : step.num}
              </div>
              <span
                className={`text-sm font-medium transition-colors ${
                  isCompleted || isCurrent ? 'text-charcoal-blue' : 'text-muted-teal'
                }`}
              >
                {step.label}
              </span>
            </div>
            
            {index < steps.length - 1 && (
              <div className="w-12 h-px bg-muted-teal/20"></div>
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
}
