import { ReactNode, useId, useState } from "react";
import { DATASET_SHORT, DatasetId, MODEL_COLOR, MODEL_LABEL, MODEL_SHORT, ModelId } from "../data";

export function Segmented<T extends string | number>({
  options, value, onChange, label, ariaLabel,
}: {
  options: { value: T; label: ReactNode }[];
  value: T;
  onChange: (v: T) => void;
  label?: string;
  ariaLabel?: string;
}) {
  return (
    <div className="control">
      {label && <div className="control-label">{label}</div>}
      <div className="seg" role="group" aria-label={ariaLabel ?? label}>
        {options.map((o) => (
          <button key={String(o.value)} aria-pressed={o.value === value} onClick={() => onChange(o.value)}>
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

export function Toggle({ checked, onChange, children }: { checked: boolean; onChange: (v: boolean) => void; children: ReactNode }) {
  return (
    <label className="toggle">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span className="track" aria-hidden />
      <span>{children}</span>
    </label>
  );
}

export function Slider({
  label, value, min, max, step = 1, onChange, format, width,
}: {
  label: ReactNode; value: number; min: number; max: number; step?: number;
  onChange: (v: number) => void; format?: (v: number) => ReactNode; width?: number;
}) {
  const id = useId();
  return (
    <div className="slider" style={width ? { maxWidth: width } : undefined}>
      <div className="slider-head">
        <label htmlFor={id}>{label}</label>
        <b>{format ? format(value) : value}</b>
      </div>
      <input id={id} type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </div>
  );
}

/** A slider over a list of discrete values (e.g. tree counts 1, 3, 10, 30…). */
export function StepSlider<T extends number>({
  label, values, value, onChange, format,
}: { label: ReactNode; values: T[]; value: T; onChange: (v: T) => void; format?: (v: T) => ReactNode }) {
  const idx = Math.max(0, values.indexOf(value));
  return (
    <Slider label={label} min={0} max={values.length - 1} value={idx} onChange={(i) => onChange(values[i])} format={() => (format ? format(value) : value)} />
  );
}

export function Card({ title, sub, chip, children, style, className }: { title?: ReactNode; sub?: ReactNode; chip?: ReactNode; children: ReactNode; style?: React.CSSProperties; className?: string }) {
  return (
    <div className={`card ${className ?? ""}`} style={style}>
      {chip ? (
        <div className="card-head" style={{ marginBottom: sub ? 4 : 10 }}>
          {title ? <div className="card-title">{title}</div> : <span />}
          {chip}
        </div>
      ) : (
        title && <div className="card-title">{title}</div>
      )}
      {sub && <div className="card-sub">{sub}</div>}
      {children}
    </div>
  );
}

/** Every card says where its numbers come from: the hypothetical Model A, or a real dataset. */
export function DataChip({ n, unit }: { n?: number; unit?: string }) {
  return n == null ? (
    <span className="data-chip hypo" title="Model A is a made-up model, built to match the lecture slides">hypothetical · 100 patients</span>
  ) : (
    <span className="data-chip real">real data · {n.toLocaleString("en-US")} {unit}</span>
  );
}

export function Stat({ label, value, note, color, huge }: { label: ReactNode; value: ReactNode; note?: ReactNode; color?: string; huge?: boolean }) {
  return (
    <div className="card">
      <div className="stat-label">{label}</div>
      <div className={`stat-value tnum ${huge ? "huge" : ""}`} style={color ? { color } : undefined}>{value}</div>
      {note && <div className="stat-note">{note}</div>}
    </div>
  );
}

export function PauseThink({ prompt, children, label = "Pause and think" }: { prompt: ReactNode; children: ReactNode; label?: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="pause">
      <div className="pause-head">
        <div>
          <div className="pause-tag">{label}</div>
          <div style={{ fontWeight: 550 }}>{prompt}</div>
        </div>
        <button className="btn" onClick={() => setOpen(!open)} aria-expanded={open}>
          {open ? "Hide" : "Reveal"}
        </button>
      </div>
      {open && <div className="pause-body">{children}</div>}
    </div>
  );
}

export function Callout({ children, tone }: { children: ReactNode; tone?: "warn" | "good" }) {
  return <div className={`callout ${tone ?? ""}`}>{children}</div>;
}

export function ModelName({ id, short }: { id: ModelId; short?: boolean }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
      <span className="swatch" style={{ background: MODEL_COLOR[id] }} />
      {short ? MODEL_SHORT[id] : MODEL_LABEL[id]}
    </span>
  );
}

export function ModelPicker({ value, onChange, models, label = "Model", short }: { value: ModelId; onChange: (m: ModelId) => void; models: ModelId[]; label?: string; short?: boolean }) {
  return <Segmented label={label} value={value} onChange={onChange} options={models.map((m) => ({ value: m, label: <ModelName id={m} short={short} /> }))} />;
}

export function DatasetPicker({ value, onChange, ids, label = "Dataset" }: { value: DatasetId; onChange: (d: DatasetId) => void; ids: DatasetId[]; label?: string }) {
  return <Segmented label={label} value={value} onChange={onChange} options={ids.map((d) => ({ value: d, label: DATASET_SHORT[d] }))} />;
}

export function Loading() {
  return <div className="loading">Loading results…</div>;
}

export function SourceNote({ children }: { children: ReactNode }) {
  return <p className="small muted" style={{ marginTop: 10 }}>{children}</p>;
}
