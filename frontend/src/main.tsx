import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, NavLink, Route, Routes } from "react-router-dom";
import { useApi, type Meta } from "./api";
import { Method } from "./pages/Method";
import { ModelLab } from "./pages/ModelLab";
import { Player } from "./pages/Player";
import { Players } from "./pages/Players";
import { Rankings } from "./pages/Rankings";
import "./styles.css";

function App() {
  const meta = useApi<Meta>("/api/meta");
  return (
    <>
      <header className="site">
        <div className="inner">
          <NavLink to="/" className="brand">🏈 QB Ratings</NavLink>
          <nav>
            <NavLink to="/" end>Rankings</NavLink>
            <NavLink to="/players">Players</NavLink>
            <NavLink to="/lab">Model Lab</NavLink>
            <NavLink to="/method">Method</NavLink>
          </nav>
        </div>
      </header>
      <main>
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
        Data: nflverse / nflfastR (CC-BY 4.0).
        {meta.data && <> Updated {new Date(meta.data.updated_at).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" })}.</>}
      </footer>
    </>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>,
);
