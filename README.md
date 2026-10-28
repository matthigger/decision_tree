# decision_tree

Interactive teaching demo of decision trees and random forests, a
companion to [KNN, visually](https://github.com/matthigger/knn). Two
tabs:

- **Decision tree.** The data picker above the plot holds two groups.
  Classification sets (Blobs, XOR, Diagonal, Moons, Rings) put samples in
  the plane, with Gini; regression sets (Sine, Step, Linear, Chirp) plot y
  against x, with MSE, a staircase of leaf means and gray sticks for the
  selected node's errors. Click a node in the tree diagram (or a region of
  the plot), split it, and drag the threshold on the plot or on a chart of
  the impurity after the split against the threshold. The readout shows
  the impurity before and after, each side's, and the arithmetic. Best
  split takes the greedy choice; Grow one level gives every leaf its best
  split. Each mode keeps its own tree.
- **Random forest.** 200 deep trees, each on its own bootstrap bag and
  each split searching one random feature; the plot shades the majority
  vote, washed out where the trees disagree. Thumbnails show the first 12
  trees with their bags (click one to see it alone), pointing at the plot
  shows the vote there, and a chart tracks the testing error (on 1,200
  fresh samples) against the number of trees, beside one tree grown on
  all the samples. Toggles turn off bagging and the random feature.

Dragging a sample re-scores the whole tree (builder) or regrows the forest.
Resample keeps the hand-built tree, to show how it does on fresh samples.

Plain HTML/CSS/JS with SVG: no build step and no dependencies.

## Run locally

Open `index.html` in a browser, or serve it:

```sh
python3 -m http.server 8000   # then visit http://localhost:8000
```

Link straight to the tree tab with `#classification` or `#regression`
(which data group it opens on), or to `#forest`.

## Publish on GitHub Pages

Pages is built by `.github/workflows/pages.yml` on every push to `main`. It
stamps `js/version.js` with the commit and build time, which the footer
shows ("local copy" when run locally). One-time setup:

```sh
gh repo create matthigger/decision_tree --public --source . --push
gh api -X POST repos/matthigger/decision_tree/pages -f build_type=workflow
```

The site then lives at <https://matthigger.github.io/decision_tree/>.

## Layout

| file | role |
|------|------|
| `index.html` | page, explanation text, controls |
| `style.css` | layout, class palette, plot and diagram styles |
| `js/data.js` | seeded data sets (classification and regression) |
| `js/tree.js` | Gini / MSE, threshold sweep, greedy fit, importance |
| `js/ui.js` | shared plot geometry, axes, formatting, build stamp |
| `js/builder.js` | classification and regression tabs (tree builder) |
| `js/forest.js` | random forest tab |
| `js/main.js` | tab switching |
| `js/version.js` | build stamp, overwritten at deploy |
| `.github/workflows/pages.yml` | Pages deploy |
