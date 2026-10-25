import { Info, X } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * Renders the official regulatory & scientific advisory disclaimer.
 * Clearly framed as a validation notice rather than a system error.
 */
export function Disclaimer({
  text,
  className,
  onDismiss,
}: {
  text: string | undefined;
  className?: string;
  onDismiss?: () => void;
}) {
  return (
    <div
      role="note"
      className={cn(
        "flex items-start justify-between gap-2.5 rounded-lg border border-sky-200/90 bg-sky-50/95 p-2.5 text-xs text-sky-950 shadow-sm backdrop-blur-sm",
        className,
      )}
    >
      <div className="flex items-start gap-2">
        <Info className="mt-0.5 h-4 w-4 shrink-0 text-sky-600" />
        <div className="space-y-0.5">
          <span className="block text-[10px] font-semibold uppercase tracking-wider text-sky-900">
            Advisory Notice
          </span>
          <span className="block leading-relaxed text-sky-950/90">
            {text ?? "Satellite-observed anomaly. Ground and laboratory testing recommended for validation."}
          </span>
        </div>
      </div>
      {onDismiss && (
        <button
          type="button"
          onClick={onDismiss}
          className="shrink-0 rounded p-1 text-sky-700/70 transition-colors hover:bg-sky-200/50 hover:text-sky-900"
          title="Dismiss advisory notice"
          aria-label="Dismiss advisory notice"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      )}
    </div>
  );
}

