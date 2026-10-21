import { useEffect, useState } from "react";
import { ALIASES, SECTIONS } from "./sections";

/** `#/threshold` or the slides' numeric form `#/6` (= the sixth section). */
function currentId() {
  const raw = window.location.hash.replace(/^#\/?/, "");
  if (/^\d+$/.test(raw)) {
    const i = Number(raw) - 1;
    if (i >= 0 && i < SECTIONS.length) return SECTIONS[i].id;
  }
  const id = ALIASES[raw] ?? raw;
  return SECTIONS.some((s) => s.id === id) ? id : SECTIONS[0].id;
}

type Theme = "auto" | "light" | "dark";

function readTheme(): Theme {
  try {
    return (localStorage.getItem("theme") as Theme) || "auto";
  } catch {
    return "auto";
  }
}

export default function App() {
  const [id, setId] = useState(currentId);
  const [menuOpen, setMenuOpen] = useState(false);
  const [theme, setTheme] = useState<Theme>(readTheme);
  const [presenter, setPresenter] = useState(false);

  useEffect(() => {
    const onHash = () => {
      setId(currentId());
      window.scrollTo({ top: 0 });
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    if (theme === "auto") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem("theme", theme);
    } catch {
      /* storage unavailable: theme still applies for this visit */
    }
  }, [theme]);

  useEffect(() => {
    document.body.classList.toggle("presenter", presenter);
  }, [presenter]);

  const idx = SECTIONS.findIndex((s) => s.id === id);
  const go = (i: number) => {
    if (i < 0 || i >= SECTIONS.length) return;
    window.location.hash = `/${SECTIONS[i].id}`;
    setMenuOpen(false);
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as Element;
      // sliders (native or the SVG threshold handles) use the arrow keys themselves
      if (t.closest?.('input, textarea, select, [contenteditable], [role="slider"]')) return;
      if (e.key === "ArrowRight" || e.key === "PageDown") go(idx + 1);
      else if (e.key === "ArrowLeft" || e.key === "PageUp") go(idx - 1);
      else if (e.key === "p") setPresenter((v) => !v);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const section = SECTIONS[idx];
  const Body = section.Component;
  let lastPart = "";

  return (
    <div className="shell">
      <nav className={`sidebar ${menuOpen ? "open" : ""}`} aria-label="Sections">
        <div className="brand">Classification Models</div>
        <div className="brand-sub">From a probability to a decision</div>
        {SECTIONS.map((s, i) => {
          const head = s.part !== lastPart ? <div className="nav-part">{s.part}</div> : null;
          lastPart = s.part;
          return (
            <div key={s.id}>
              {head}
              <button className={`nav-item ${i === idx ? "active" : ""}`} onClick={() => go(i)} aria-current={i === idx ? "page" : undefined}
                title={`Slide${s.slides.includes("–") ? "s" : ""} ${s.slides}`}>
                <span className="nav-num">{i + 1}</span>
                <span>{s.title}</span>
                {s.tryIt && <span className="nav-try" title="Homework: the Try it prompts in this section">Try it</span>}
              </button>
            </div>
          );
        })}
        <div className="sidebar-foot">
          <kbd>←</kbd> <kbd>→</kbd> move between sections · <kbd>p</kbd> presenter mode
          <br />
          Homework: <b>Try it</b> in §2, §6 and §8
          <br />
          <a href="#/checklist" onClick={() => setMenuOpen(false)}>Data sources &amp; reproducibility</a>
        </div>
      </nav>

      <div className="main">
        <header className="topbar">
          <button className="btn menu-btn" onClick={() => setMenuOpen(!menuOpen)} aria-label="Open section menu">☰</button>
          <div className="small muted tnum" style={{ whiteSpace: "nowrap" }}>
            {idx + 1} / {SECTIONS.length}
          </div>
          <div className="progress" aria-hidden>
            <div style={{ width: `${((idx + 1) / SECTIONS.length) * 100}%` }} />
          </div>
          <div className="topbar-actions">
            <button className="btn" onClick={() => setPresenter(!presenter)} aria-pressed={presenter} title="Larger text for recording (p)">
              {presenter ? "Exit presenter" : "Presenter"}
            </button>
            <button
              className="btn"
              onClick={() => setTheme(theme === "auto" ? "dark" : theme === "dark" ? "light" : "auto")}
              title="Theme: auto → dark → light"
            >
              {theme === "auto" ? "Auto theme" : theme === "dark" ? "Dark" : "Light"}
            </button>
          </div>
        </header>

        <main className="stage" key={section.id}>
          <div className="section-kicker">
            {section.part} · {idx + 1} <span className="kicker-slides">· slide{section.slides.includes("–") ? "s" : ""} {section.slides}</span>
          </div>
          <h1>{section.title}</h1>
          {section.lede && <p className="lede">{section.lede}</p>}
          <Body />
          <div className="pager">
            <button className="btn" onClick={() => go(idx - 1)} disabled={idx === 0} style={{ visibility: idx === 0 ? "hidden" : undefined }}>
              ← {idx > 0 ? SECTIONS[idx - 1].title : ""}
            </button>
            {idx < SECTIONS.length - 1 && (
              <button className="btn primary" onClick={() => go(idx + 1)}>
                {SECTIONS[idx + 1].title} →
              </button>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
