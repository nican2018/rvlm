'use client';

import { useEffect, useState } from 'react';

// RVLM Architecture ASCII art — vision-aware recursive loop
const RVLM_SIMPLE = `
  ┌──────────┐      ╔══════════════════════════════════════════════╗      ┌──────────┐
  │  Prompt  │      ║              RVLM (depth=0)                  ║      │  Answer  │
  │──────────│ ───► ║  ┌──────────────────────────────────────┐    ║ ───► │──────────│
  │ context  │      ║  │     Vision Language Model (VLM)      │    ║      │  FINAL() │
  │ images[] │      ║  └──────────────────┬───────────────────┘    ║      └──────────┘
  └──────────┘      ║                    ↓ ↑                       ║
                    ║  ┌──────────────────▼───────────────────┐    ║
                    ║  │         Environment (REPL)            │    ║
                    ║  │  describe_image() · llm_query()       │    ║
                    ║  │  llm_query_with_images()              │    ║
                    ║  └───────────┬───────────┬───────────────┘    ║
                    ╚══════════════│═══════════│════════════════════╝
                                   │           │
                          ┌────────▼────┐ ┌────▼────────┐
                          │ sub-VLM     │ │ sub-VLM     │
                          │ with images │ │ with images │
                          └────────┬────┘ └────┬────────┘
                                   │           │
                          ╔════════▼════╗ ╔════▼════════╗
                          ║ RVLM (d=1)  ║ ║ RVLM (d=1)  ║
                          ║  VLM ↔ REPL ║ ║  VLM ↔ REPL ║
                          ╚═════════════╝ ╚═════════════╝
`;

export function AsciiRVLM() {
  const [pulse, setPulse] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setPulse(p => (p + 1) % 4);
    }, 600);
    return () => clearInterval(interval);
  }, []);

  // Colorize the ASCII art
  const colorize = (text: string) => {
    return text.split('\n').map((line, lineIdx) => (
      <div key={lineIdx} className="whitespace-pre">
        {line.split('').map((char, charIdx) => {
          const key = `${lineIdx}-${charIdx}`;
          
          // Box drawing characters - dim
          if ('┌┐└┘├┤┬┴┼─│╔╗╚╝║═'.includes(char)) {
            return <span key={key} className="text-muted-foreground/50">{char}</span>;
          }
          // Arrows - primary color
          if ('▼▲↓↑→←'.includes(char)) {
            const isPulsing = (lineIdx + charIdx + pulse) % 4 === 0;
            return (
              <span 
                key={key} 
                className={isPulsing ? 'text-primary' : 'text-primary/60'}
              >
                {char}
              </span>
            );
          }
          // Keywords
          if (line.includes('RVLM') && char !== ' ') {
            if ('RVLM'.includes(char)) {
              return <span key={key} className="text-primary font-bold">{char}</span>;
            }
          }
          if (line.includes('Prompt') || line.includes('Response') || line.includes('Answer')) {
            if (!'[]│─'.includes(char) && char !== ' ') {
              return <span key={key} className="text-amber-600 dark:text-amber-400">{char}</span>;
            }
          }
          if (line.includes('Vision Language Model') || line.includes('VLM')) {
            if (!'[]│─┌┐└┘'.includes(char) && char !== ' ') {
              return <span key={key} className="text-sky-600 dark:text-sky-400">{char}</span>;
            }
          }
          if (line.includes('REPL') || line.includes('Environment') || line.includes('describe_image') || line.includes('llm_query')) {
            if (!'[]│─┌┐└┘'.includes(char) && char !== ' ') {
              return <span key={key} className="text-emerald-600 dark:text-emerald-400">{char}</span>;
            }
          }
          if (line.includes('images') && !line.includes('llm_query')) {
            if (!'[]│─┌┐└┘'.includes(char) && char !== ' ') {
              return <span key={key} className="text-fuchsia-600 dark:text-fuchsia-400">{char}</span>;
            }
          }
          if (line.includes('depth=') || line.includes('d=')) {
            if (!'()'.includes(char) && char !== ' ') {
              return <span key={key} className="text-muted-foreground">{char}</span>;
            }
          }
          // Default
          return <span key={key} className="text-muted-foreground/70">{char}</span>;
        })}
      </div>
    ));
  };

  return (
    <div className="font-mono text-[10px] leading-[1.3] select-none">
      <pre>{colorize(RVLM_SIMPLE)}</pre>
    </div>
  );
}

// Compact inline diagram for header
export function AsciiRVLMInline() {
  return (
    <div className="font-mono text-[9px] leading-tight select-none text-muted-foreground">
      <span className="text-fuchsia-600 dark:text-fuchsia-400">Images</span>
      <span> + </span>
      <span className="text-primary">Prompt</span>
      <span> → </span>
      <span className="text-emerald-600 dark:text-emerald-400">[VLM ↔ REPL]</span>
      <span> → </span>
      <span className="text-amber-600 dark:text-amber-400">Answer</span>
    </div>
  );
}

// Keep old export name for backward compat
export { AsciiRVLM as AsciiRLM };
