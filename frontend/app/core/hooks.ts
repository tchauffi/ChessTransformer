import { useEffect, useRef, useState } from 'react';

/** State mirrored to localStorage; storage failures (private mode, blocked) fall back to memory. */
export function useStoredState<T extends object>(key: string, initial: T) {
  const [state, setState] = useState(initial);
  const loaded = useRef(false);

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(key);
      if (raw) setState({ ...initial, ...JSON.parse(raw) });
    } catch {
      // keep defaults
    }
    loaded.current = true;
    // eslint-disable-next-line react-hooks/exhaustive-deps -- initial is a constant default
  }, [key]);

  useEffect(() => {
    if (!loaded.current) return;
    try {
      window.localStorage.setItem(key, JSON.stringify(state));
    } catch {
      // not persisted; fine
    }
  }, [key, state]);

  return [state, setState] as const;
}
