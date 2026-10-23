# decision_tree

Interactive teaching demo of decision trees and random forests, a
companion to [KNN, visually](https://github.com/matthigger/knn). Three
tabs:

- **Classification tree.** Samples in the plane (Blobs, XOR, Diagonal,
  Moons, Rings). Click a node in the tree diagram (or a region of the
  plot), split it on x1 or x2, and drag the threshold: on the plot, with
  the slider, or on a chart of the weighted Gini after the split against
  the threshold. A table shows the Gini before, on each side, the
  weighted average after and the reduction. Best split takes the greedy
  choice; Grow splits every leaf under the selected node greedily down to
  a max depth.
- **Regression tree.** The same builder for y against x (Sine, Step,
  Linear, Chirp), with MSE in place of Gini. The estimate is a staircase
  of leaf means; gray sticks show the selected node's errors.
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

Link straight to a tab with `#classification`, `#regression` or
`#forest`.

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
