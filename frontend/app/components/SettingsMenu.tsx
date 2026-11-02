'use client';

import { useEffect, useRef, useState } from 'react';
import { IconGear } from './icons';

export interface Settings {
  sound: boolean;
  legalMoves: boolean;
  evalBar: boolean;
  autoQueen: boolean;
}

export const DEFAULT_SETTINGS: Settings = { sound: true, legalMoves: true, evalBar: true, autoQueen: false };

const ITEMS: { key: keyof Settings; label: string }[] = [
  { key: 'sound', label: 'Sound' },
  { key: 'legalMoves', label: 'Show legal moves' },
  { key: 'evalBar', label: 'Evaluation bar' },
  { key: 'autoQueen', label: 'Always promote to queen' },
];

export default function SettingsMenu({ settings, onChange }: { settings: Settings; onChange: (s: Settings) => void }) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent | KeyboardEvent) => {
      if (e instanceof KeyboardEvent ? e.key === 'Escape' : !root.current?.contains(e.target as Node)) setOpen(false);
    };
    window.addEventListener('mousedown', close);
    window.addEventListener('keydown', close);
    return () => {
      window.removeEventListener('mousedown', close);
      window.removeEventListener('keydown', close);
    };
  }, [open]);

  return (
    <div ref={root} className="relative">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-label="Settings"
        className="rounded-md p-2 text-muted hover:bg-raised hover:text-fg"
      >
        <IconGear />
      </button>
      {open ? (
        <div className="absolute right-0 top-full z-50 mt-1 w-64 rounded-lg border border-line bg-raised p-1.5 shadow-[0_12px_32px_rgba(0,0,0,0.45)]">
          {ITEMS.map(({ key, label }) => (
            <label
              key={key}
              className="flex cursor-pointer items-center justify-between rounded-md px-2.5 py-2 text-sm text-fg hover:bg-line/60"
            >
              {label}
              <input
                type="checkbox"
                className="switch"
                checked={settings[key]}
                onChange={() => onChange({ ...settings, [key]: !settings[key] })}
              />
            </label>
          ))}
        </div>
      ) : null}
    </div>
  );
}
