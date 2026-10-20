import { createContext, useContext, useEffect, useState, type ReactNode } from "react";

export type Theme = "dark" | "light";

// Dark is the default; the choice is remembered per browser (storage can be blocked, so every access is guarded).
const KEY = "qb-theme";
const read = (): Theme => {
  try {
    return localStorage.getItem(KEY) === "light" ? "light" : "dark";
  } catch {
    return "dark";
  }
};

const ThemeContext = createContext<{ theme: Theme; toggle: () => void }>({ theme: "dark", toggle: () => {} });

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<Theme>(read);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem(KEY, theme);
    } catch {
      /* private mode: the toggle still works for this visit */
    }
  }, [theme]);
  return <ThemeContext.Provider value={{ theme, toggle: () => setTheme((t) => (t === "dark" ? "light" : "dark")) }}>{children}</ThemeContext.Provider>;
}

export const useTheme = () => useContext(ThemeContext);

// Chart surfaces (keep in sync with --surface-1 in styles.css).
const SURFACE: Record<Theme, string> = { dark: "#141820", light: "#ffffff" };

const rgb = (hex: string) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
const luminance = (c: number[]) => {
  const [r, g, b] = c.map((v) => {
    const s = v / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};
const contrast = (a: number[], b: number[]) => {
  const [x, y] = [luminance(a), luminance(b)].sort((m, n) => n - m);
  return (x + 0.05) / (y + 0.05);
};

/**
 * A team's primary color, nudged just enough to stand out from the chart surface. Several primaries are near-black
 * navies (BAL, CHI, NE...) that would vanish on the dark theme; they're blended toward white until they reach 2:1.
 */
export function markColor(hex: string | undefined, theme: Theme): string {
  if (!hex || !/^#[0-9a-f]{6}$/i.test(hex)) return "#8a909c";
  const surface = rgb(SURFACE[theme]);
  const toward = theme === "dark" ? [255, 255, 255] : [0, 0, 0];
  let c = rgb(hex);
  for (let i = 0; i < 12 && contrast(c, surface) < 2; i++) c = c.map((v, k) => Math.round(v + (toward[k] - v) * 0.12));
  return `#${c.map((v) => v.toString(16).padStart(2, "0")).join("")}`;
}
