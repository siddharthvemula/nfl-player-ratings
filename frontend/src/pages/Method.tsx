import { useApi, type Meta } from "../api";
import { FEATURE_LABELS, signed, TYPICAL_DROPBACKS } from "../format";

export function Method() {
  const meta = useApi<Meta>("/api/meta");
  const model = meta.data?.models[meta.data.models.length - 1];
  const avg = model?.league_avg;
  return (
    <div className="prose">
      <div className="eyebrow">Method</div>
      <h1>How the ratings work</h1>

      <h2>What a rating is</h2>
      <p>
        A QB's rating is a forecast: his <strong>EPA per dropback</strong> in his next game. EPA (expected points added) measures how much
        each play changed his team's expected score, so a 3rd-and-15 conversion counts for more than a garbage-time checkdown.
        Dropbacks include passes, sacks and scrambles.
      </p>
      <div className="explain" style={{ marginBottom: 16 }}>
        <span className="i" aria-hidden="true">i</span>
        <span>
          <strong>Reading the number.</strong> A rating of +0.24 means he's expected to add 0.24 points to his team's expected score on
          each dropback. Zero is <em>not</em> average: the average QB adds about {avg ? signed(avg, 2) : "+0.06"} per dropback, so +0.24 is
          about {avg ? (0.24 - avg).toFixed(2) : "0.18"} better than average. Over a typical {TYPICAL_DROPBACKS}-dropback game, that's
          roughly {avg ? ((0.24 - avg) * TYPICAL_DROPBACKS).toFixed(1) : "6.5"} more points than an average QB would produce.
        </span>
      </div>
      <p>Because a rating is a forecast, it can be checked: after each game, the site shows what the QB actually did.</p>

      <h2>How it's made</h2>
      <ol>
        <li>
          <strong>Carry-over.</strong> Start from last season's EPA/dropback, pulled toward the league average when it was based on few
          dropbacks. As this season's dropbacks pile up, they take over. This alone beats "league average for everyone."
        </li>
        <li>
          <strong>A learned adjustment</strong> on top, from six features: {Object.values(FEATURE_LABELS).map((l) => l[0].toLowerCase() + l.slice(1)).join(", ")}.
          It's a weighted sum, so every rating splits exactly into one piece per feature (the breakdown on each player page).
        </li>
        <li>
          <strong>No peeking.</strong> A rating for week N only uses games before week N. Each season is rated by a model trained only on
          earlier seasons, so every rating on this site was made before the game it predicts.
        </li>
      </ol>
      <p>
        Two findings worth knowing. QBs with little history (rookies, backups) do worse than carry-over assumes, so the model marks them down.
        And at the same past EPA, a QB who took <em>more</em> sacks tends to do slightly better next week: sacks are costly but don't repeat
        as reliably as the rest of a QB's play, so carry-over over-punishes them.
      </p>

      <h2>How it was tested</h2>
      <p>
        Ideas were tested on six held-out seasons (2019-2024), each trained only on earlier seasons, and kept only if they won clearly.
        Home field, completion % over expected, Ridge regression and gradient-boosted trees (LightGBM) were all tried and dropped:
        none beat the simple model. The final model was then frozen and tested once on 2025. See the Model Lab for every experiment.
      </p>

      {model && (
        <>
          <h2>The current model ({model.season})</h2>
          <p className="secondary">Trained on {model.trained_on}. Weight per unit of each feature (applied to his value minus the training average):</p>
          <div className="table-wrap" style={{ maxWidth: 560 }}>
            <table>
              <thead><tr><th>Feature</th><th className="num">Weight</th><th className="num">Training average</th></tr></thead>
              <tbody>
                {Object.entries(model.weights).map(([f, w]) => (
                  <tr key={f}><td>{FEATURE_LABELS[f] ?? f}</td><td className="num">{w.toFixed(4)}</td><td className="num">{model.feature_means[f].toFixed(3)}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      <h2>Limitations</h2>
      <ul>
        <li>EPA and CPOE come from nflverse's own models, which were fit on seasons that overlap the ones used here. Every feature and the target share that small risk.</li>
        <li>2025 has been used as the one-time test, so later model changes can only be judged on past seasons.</li>
        <li>The model knows nothing about injuries, depth charts, weather or coaching changes. It doesn't know who will start.</li>
        <li>The gain over carry-over comes almost entirely from QBs with little history. For established starters, the rating is close to carry-over.</li>
        <li>One game is noisy: even the best forecasts miss a single game by about 0.2 EPA/dropback on average.</li>
      </ul>

      <h2>Data</h2>
      <p>
        Data: <a href="https://github.com/nflverse">nflverse</a> / nflfastR (CC-BY 4.0), via nflreadpy. Ratings are recomputed every Tuesday.
      </p>
    </div>
  );
}
