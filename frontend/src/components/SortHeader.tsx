export type SortState<K extends string> = { key: K; dir: "asc" | "desc" };

/** A clickable column header: first click sorts by the column's natural direction, the next flips it. */
export function SortHeader<K extends string>({ label, k, sort, onSort, num, className, title, natural = "desc" }: {
  label: string; k: K; sort: SortState<K>; onSort: (s: SortState<K>) => void; num?: boolean; className?: string; title?: string; natural?: "asc" | "desc";
}) {
  const active = sort.key === k;
  const next = () => onSort({ key: k, dir: active ? (sort.dir === "asc" ? "desc" : "asc") : natural });
  return (
    <th className={`${num ? "num" : ""} ${className ?? ""}`} aria-sort={active ? (sort.dir === "asc" ? "ascending" : "descending") : "none"}>
      <button onClick={next} title={title}>
        {label}
        <span className="arrow" aria-hidden="true">{active ? (sort.dir === "asc" ? "▲" : "▼") : ""}</span>
      </button>
    </th>
  );
}
