// Tabs: the tree builder (classification or regression) and the random
// forest. Each tab keeps its own data and tree while hidden.

const state = { mode: "class" };
const HASH = { class: "#classification", reg: "#regression",
  forest: "#forest" };

function setMode(mode) {
  state.mode = mode;
  history.replaceState(null, "", HASH[mode]);
  for (const t of document.querySelectorAll(".tab")) {
    t.setAttribute("aria-selected", t.dataset.mode === mode);
  }
  for (const el of document.querySelectorAll("[data-show]")) {
    el.hidden = !el.dataset.show.split(" ").includes(mode);
  }
  if (mode === "forest") forestShow();
  else builderShow(mode);
}

for (const t of document.querySelectorAll(".tab")) {
  t.onclick = () => {
    if (t.dataset.mode !== state.mode) setMode(t.dataset.mode);
  };
}

setMode(Object.keys(HASH).find(m => HASH[m] === location.hash) || "class");
