import { Analytics } from "@vercel/analytics/react";
import React, { useEffect } from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, NavLink, Route, Routes, useLocation } from "react-router-dom";
import { useApi, type Meta } from "./api";
import { Method } from "./pages/Method";
import { ModelLab } from "./pages/ModelLab";
import { Player } from "./pages/Player";
import { Players } from "./pages/Players";
import { Rankings } from "./pages/Rankings";
import { ThemeProvider, useTheme } from "./theme";
import "./styles.css";

function ThemeToggle() {
  const { theme, toggle } = useTheme();
  const next = theme === "dark" ? "light" : "dark";
  return (
    <button className="theme-toggle" onClick={toggle} aria-label={`Switch to ${next} theme`}>
      <span aria-hidden="true">{theme === "dark" ? "☀" : "☾"}</span>
      <span className="label">{theme === "dark" ? "Light" : "Dark"}</span>
    </button>
  );
}

function App() {
  const meta = useApi<Meta>("/api/meta");
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);
  return (
    <>
      <header className="site">
        <div className="inner">
          <NavLink to="/" className="brand"><span className="brand-mark" aria-hidden="true">QB</span>QB Ratings</NavLink>
          <nav>
            <NavLink to="/" end>Rankings</NavLink>
            <NavLink to="/players">Players</NavLink>
            <NavLink to="/lab">Model Lab</NavLink>
            <NavLink to="/method">Method</NavLink>
          </nav>
          <ThemeToggle />
        </div>
      </header>
      {/* key: replay the page-in animation on navigation */}
      <main key={pathname}>
        <Routes>
          <Route path="/" element={<Rankings />} />
          <Route path="/players" element={<Players />} />
          <Route path="/player/:id" element={<Player />} />
          <Route path="/lab" element={<ModelLab />} />
          <Route path="/method" element={<Method />} />
          <Route path="*" element={<p>Page not found.</p>} />
        </Routes>
      </main>
      <footer>
        Data: <a href="https://github.com/nflverse">nflverse</a> / nflfastR (CC-BY 4.0).
        {meta.data && <> Updated {new Date(meta.data.updated_at).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" })}.</>}
      </footer>
    </>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ThemeProvider>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </ThemeProvider>
    {/* Vercel Web Analytics: page views only, no cookies. Does nothing until enabled in the Vercel project. */}
    <Analytics />
  </React.StrictMode>,
);
