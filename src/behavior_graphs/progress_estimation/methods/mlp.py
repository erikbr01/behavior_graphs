"""Direct-regression MLP progress method.

A simple learned baseline: a small MLP that regresses per-frame progress
DIRECTLY from a policy embedding, in contrast to the graph (which reads progress
off cluster-transition structure) and the VLMs (which score video). It is the
learned analog of WARP-RM's premise — dense progress supervised on successful
demos — but on the policy's own representation rather than raw pixels.

``fit`` trains a StandardScaler + MLPRegressor on the TRAIN split; the target is
normalized time ``linspace(0, 1, T)`` over SUCCESSFUL train episodes only (a
failed demo never "completes", so a full 0->1 ramp target would teach the head
that failures finish -> hurts success separation). ``predict`` regresses each
held-out episode's frames. An optional temporal window concatenates the previous
``window_width - 1`` frames (``window_width=1`` = pure per-frame, the simplest
baseline). Two backends: sklearn on CPU (the default) or torch on GPU, which is
what makes a full-store fit tractable -- see ``MLPRegressorProgress``.
"""

from __future__ import annotations

import numpy as np

from behavior_graphs.progress_estimation.methods.base import ProgressMethod


def _stack_window(ep: np.ndarray, width: int) -> np.ndarray:
    """Per-frame feature = concat of the current and previous ``width - 1`` frames
    (left-padded with the first frame). ``width <= 1`` -> identity (per-frame)."""
    if width <= 1:
        return ep
    T, d = ep.shape
    out = np.empty((T, d * width), dtype=ep.dtype)
    for j in range(width):
        shifted = np.empty_like(ep)
        if j:
            # `width > T + 1` is reachable -- window_width is a config knob and
            # the benchmark spans 6-to-708-step episodes -- and once j > T the
            # unguarded `shifted[j:] = ep[:T - j]` broadcast a negative slice
            # and raised mid-fit. Past the start of the episode there is
            # nothing but the left-pad.
            keep = max(T - j, 0)
            if keep:
                shifted[j:] = ep[:keep]
            shifted[: T - keep] = ep[0]
        else:
            shifted[:] = ep
        out[:, j * d : (j + 1) * d] = shifted
    return out


class MLPRegressorProgress(ProgressMethod):
    """MLP that regresses per-frame progress from a policy embedding.

    Two interchangeable backends fit the same model. ``backend="sklearn"`` is the
    original and stays the default so existing results reproduce.
    ``backend="torch"`` exists because sklearn does not scale to a full store:
    on 2.2M frames x 512 dims it is ~11k minibatches per epoch of 200 over a
    4.5 GB matrix, single-process, and a fit ran 2h45m without completing once.
    Subsampling to make it terminate is not a safe workaround -- fitting on 400 vs
    800 episodes moved the top-28.9% kept-frame set by 12% (Jaccard 0.78), and for
    a curation experiment *which frames are kept* is the whole independent
    variable.

    The torch path keeps the objective, the L2 convention, and the early-stopping
    rule identical to sklearn's ``MLPRegressor``; it deliberately differs in
    minibatch size (200 is launch-bound on a GPU) and so is not bit-comparable.
    Measured on identical episodes: correlation with the regression target 0.9579
    (torch) vs 0.9585 (sklearn), and 93.6% agreement on the top-28.9% kept-frame
    set -- a smaller perturbation than the subsampling it removes. Re-measure that
    way after changing either path; do not assume equivalence.

    Features stay in host memory and stream to the device per minibatch. Holding
    the full matrix on the GPU would add ~4.5 GB, which is enough to OOM a
    co-resident training run -- and this fit is never the critical path.
    """

    modality = "embedding"
    needs_backbone = False

    def __init__(
        self,
        name: str = "mlp",
        window_width: int = 1,
        hidden_layer_sizes: tuple[int, ...] = (256,),
        alpha: float = 1e-3,
        max_iter: int = 400,
        seed: int = 0,
        backend: str = "sklearn",
        device: str = "cuda",
        batch_size: int = 4096,
        verbose: bool = False,
    ):
        if backend not in ("sklearn", "torch"):
            raise ValueError(f"backend must be 'sklearn' or 'torch', got {backend!r}")
        self.name = name
        self.window_width = window_width
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.alpha = alpha
        self.max_iter = max_iter
        self.seed = seed
        self.backend = backend
        self.device = device
        self.batch_size = batch_size
        self.verbose = verbose
        self._scaler = None
        self._mlp = None
        self._torch = None  # (module, mean, std) once fitted

    def reset(self) -> None:
        self._scaler = self._mlp = self._torch = None

    def _design(self, train):
        """Stacked features and normalized-time targets over successful episodes."""
        xs, ys = [], []
        for ep, ok in zip(train.embeddings(), train.flags):
            if not ok:  # supervise on successful demos only
                continue
            w = _stack_window(np.asarray(ep, dtype=np.float32), self.window_width)
            xs.append(w)
            ys.append(np.linspace(0.0, 1.0, len(w), dtype=np.float32))
        if not xs:
            raise ValueError(f"{self.name}: no successful train episodes to fit on")
        return np.concatenate(xs, 0), np.concatenate(ys, 0)

    def fit(self, train) -> None:
        x, y = self._design(train)
        if self.backend == "torch":
            self._fit_torch(x, y)
        else:
            self._fit_sklearn(x, y)

    def _fit_sklearn(self, x, y) -> None:
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler

        self._scaler = StandardScaler().fit(x)
        self._mlp = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            alpha=self.alpha,
            max_iter=self.max_iter,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=self.seed,
        )
        self._mlp.fit(self._scaler.transform(x), y)

    def _build(self, d_in: int):
        import torch

        sizes = [d_in, *self.hidden_layer_sizes]
        layers = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            layers += [torch.nn.Linear(a, b), torch.nn.ReLU()]
        layers.append(torch.nn.Linear(sizes[-1], 1))
        return torch.nn.Sequential(*layers)

    def _fit_torch(self, x, y) -> None:
        """Same objective and stopping rule as sklearn, minibatched on the device.

        Held to sklearn's conventions on purpose: squared loss halved, L2 on
        weights only (not biases) scaled by 1/batch, Adam at its sklearn defaults,
        a 10% validation split, and stop when R^2 fails to improve by ``tol`` for
        15 consecutive epochs -- restoring the best weights, not the last ones.
        """
        import torch

        torch.manual_seed(self.seed)
        dev = torch.device(self.device if torch.cuda.is_available() else "cpu")

        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(x))
        n_val = max(1, int(0.1 * len(x)))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]

        # Standardize from train rows only, then normalize per minibatch on the
        # device. StandardScaler.transform would instead materialize a second full
        # copy of x -- 4.5 GB on this store, and the reason RSS hit 11.7 GB.
        mean = x[tr_idx].mean(0)
        std = x[tr_idx].std(0)
        std[std < 1e-8] = 1.0
        t_mean = torch.from_numpy(mean).to(dev)
        t_std = torch.from_numpy(std).to(dev)

        model = self._build(x.shape[1]).to(dev)
        weights = [m.weight for m in model if isinstance(m, torch.nn.Linear)]
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)

        xv = torch.from_numpy(np.ascontiguousarray(x[val_idx])).to(dev)
        yv = torch.from_numpy(np.ascontiguousarray(y[val_idx])).to(dev)
        xv = (xv - t_mean) / t_std
        v_var = torch.var(yv, unbiased=False)

        x_tr, y_tr = x[tr_idx], y[tr_idx]
        best, best_state, stale = -np.inf, None, 0
        for epoch in range(self.max_iter):
            model.train()
            order = rng.permutation(len(x_tr))
            for b0 in range(0, len(order), self.batch_size):
                sel = order[b0 : b0 + self.batch_size]
                xb = torch.from_numpy(np.ascontiguousarray(x_tr[sel])).to(dev)
                yb = torch.from_numpy(np.ascontiguousarray(y_tr[sel])).to(dev)
                pred = model((xb - t_mean) / t_std).squeeze(-1)
                l2 = sum((w * w).sum() for w in weights)
                loss = 0.5 * torch.mean((pred - yb) ** 2) + 0.5 * self.alpha * l2 / len(sel)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                pv = model(xv).squeeze(-1)
                r2 = float(1.0 - torch.mean((pv - yv) ** 2) / v_var)
            if r2 > best + 1e-4:  # sklearn's tol
                best, stale = r2, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if self.verbose:
                print(
                    f"  epoch {epoch + 1} val_r2={r2:.5f} best={best:.5f} stale={stale}", flush=True
                )
            if stale >= 15:  # sklearn's n_iter_no_change
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self._torch = (model, t_mean, t_std)
        if self.verbose:
            print(f"  fit done: {epoch + 1} epochs, best val_r2={best:.5f}", flush=True)

    def predict(self, eval) -> list[np.ndarray]:
        if self._mlp is None and self._torch is None:
            raise RuntimeError(f"{self.name}: fit() must run before predict()")
        if self._torch is not None:
            return self._predict_torch(eval)
        curves = []
        for ep in eval.embeddings():
            w = _stack_window(np.asarray(ep, dtype=np.float32), self.window_width)
            curves.append(np.clip(self._mlp.predict(self._scaler.transform(w)), 0.0, 1.0))
        return curves

    def _predict_torch(self, eval) -> list[np.ndarray]:
        import torch

        model, t_mean, t_std = self._torch
        curves = []
        with torch.no_grad():
            for ep in eval.embeddings():
                w = _stack_window(np.asarray(ep, dtype=np.float32), self.window_width)
                out = []
                for b0 in range(0, len(w), self.batch_size):
                    xb = torch.from_numpy(np.ascontiguousarray(w[b0 : b0 + self.batch_size]))
                    xb = (xb.to(t_mean.device) - t_mean) / t_std
                    out.append(model(xb).squeeze(-1).float().cpu().numpy())
                curves.append(np.clip(np.concatenate(out), 0.0, 1.0).astype(np.float64))
        return curves
