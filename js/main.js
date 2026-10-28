// Tabs: the tree builder and the random forest. The builder's mode
// (classification or regression) follows its data picker; each mode and
// the forest keep their own data and tree while hidden.

const state = { mode: "class" };
const HASH = { class: "#classification", reg: "#regression",
  forest: "#forest" };

// Builder mode last shown, so the tree tab reopens where it was left.
let treeMode = "class";

function tabOf(mode) { return mode === "forest" ? "forest" : "class"; }

function setMode(mode) {
  if (mode !== "forest") treeMode = mode;
  state.mode = mode;
  history.replaceState(null, "", HASH[mode]);
  for (const t of document.querySelectorAll(".tab")) {
    t.setAttribute("aria-selected", t.dataset.mode === tabOf(mode));
  }
  for (const el of document.querySelectorAll("[data-show]")) {
    el.hidden = !el.dataset.show.split(" ").includes(mode);
  }
  if (mode === "forest") forestShow();
  else builderShow(mode);
}

for (const t of document.querySelectorAll(".tab")) {
  t.onclick = () => {
    if (t.dataset.mode === tabOf(state.mode)) return;
    setMode(t.dataset.mode === "forest" ? "forest" : treeMode);
  };
}

setMode(Object.keys(HASH).find(m => HASH[m] === location.hash) || "class");
