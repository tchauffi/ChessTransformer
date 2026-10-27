import type { Metadata } from 'next';
import { notFound } from 'next/navigation';
import GameViewer from '../components/GameViewer';

export const metadata: Metadata = {
  title: 'Self-play games · ChessTransformer',
  description: "Browse the self-play games of ChessTransformer's training runs, with each move's search",
};

// The viewer reads the training machine's games through scripts/games_viewer.py (`make viewer`
// builds it with NEXT_PUBLIC_GAMES_VIEWER=1); the public Space's build leaves it out.
export default function GamesPage() {
  if (process.env.NEXT_PUBLIC_GAMES_VIEWER !== '1') notFound();
  return <GameViewer />;
}
