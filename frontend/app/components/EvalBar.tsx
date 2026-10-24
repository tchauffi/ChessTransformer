import { formatEval } from '../core/chess';

interface Props {
  /** Expected score for White in (-1, 1), or null before the engine has spoken. */
  value: number | null;
  /** Final score once the game is decided ('1-0', '0-1', '1/2-1/2'). */
  final?: string;
  orientation: 'white' | 'black';
}

export default function EvalBar({ value, final, orientation }: Props) {
  let white = value == null ? 0.5 : Math.min(0.97, Math.max(0.03, (value + 1) / 2));
  if (final === '1-0') white = 1;
  else if (final === '0-1') white = 0;
  else if (final === '1/2-1/2') white = 0.5;

  const whiteAhead = white >= 0.5;
  const label = final ?? (value == null ? '0.0' : formatEval(value));
  // The label sits at the end of the bar belonging to the side that is ahead.
  const labelAtBottom = whiteAhead === (orientation === 'white');

  return (
    <div
      className="relative h-full w-full overflow-hidden rounded-[3px] bg-[#3b3f42]"
      role="meter"
      aria-label={`Engine evaluation ${label}`}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(white * 100)}
    >
      <div
        className="absolute inset-x-0 bg-[#f1efe8] transition-[height] duration-700 ease-out motion-reduce:transition-none"
        style={{ height: `${white * 100}%`, [orientation === 'white' ? 'bottom' : 'top']: 0 }}
      />
      <span
        className={`absolute inset-x-0 text-center text-[10px] font-semibold tabular-nums max-lg:hidden ${
          labelAtBottom ? 'bottom-1' : 'top-1'
        } ${whiteAhead ? 'text-[#3b3f42]' : 'text-[#f1efe8]'}`}
      >
        {label}
      </span>
    </div>
  );
}
