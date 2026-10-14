"""An exact, faster _inputs: the same numbers as the engine's, built with fewer Python steps."""
import numpy as np


def install(cls):
    base_pre = cls._precompute_features
    base_inputs = cls._inputs

    def _precompute_features(self):
        feats = base_pre(self)
        log_deg, neighs, q_tok, q_deg, log_tok, at_risk = feats
        nodes = list(log_deg.keys())
        idx = {u: i for i, u in enumerate(nodes)}
        F = np.empty((len(nodes), 14))
        for i, u in enumerate(nodes):
            F[i, 0] = log_tok.get(u, 0.0)
            F[i, 1] = log_deg[u]
            F[i, 2:8] = q_tok[u]
            F[i, 8:14] = q_deg[u]
        self._fast = (id(log_deg), idx, F)
        return feats

    def _inputs(self, u, candidates, log_deg, q_tok, q_deg, log_tok, at_risk):
        fast = getattr(self, "_fast", None)
        if fast is None or fast[0] != id(log_deg):
            return base_inputs(self, u, candidates, log_deg, q_tok, q_deg, log_tok, at_risk)
        _, idx, F = fast
        cfg = self.cfg
        m = cfg.message_amount
        rows = getattr(self, "_rows_cache", None)
        if rows is None:
            rows = {name: slice(start, stop) for name, (start, stop, _) in cfg.input_layout().items()}
            self._rows_cache = rows
        n = len(candidates)
        X = np.empty((rows["noise"].stop, n))
        X[rows["self"]] = [u == v for v in candidates]
        if cfg.allow_gifting:
            X[rows["at_risk"]] = [u != v and self._edge_key(u, v) in at_risk for v in candidates]
        try:
            ci = [idx[v] for v in candidates]
        except KeyError:
            return base_inputs(self, u, candidates, log_deg, q_tok, q_deg, log_tok, at_risk)
        C = F[ci]                      # (n, 14)
        own = F[idx[u]]
        X[rows["own_tokens"]] = own[0]
        X[rows["its_tokens"]] = C[:, 0]
        X[rows["own_degree"]] = own[1]
        X[rows["its_degree"]] = C[:, 1]
        X[rows["own_token_quantiles"]] = own[2:8, None]
        X[rows["its_token_quantiles"]] = C[:, 2:8].T
        X[rows["own_degree_quantiles"]] = own[8:14, None]
        X[rows["its_degree_quantiles"]] = C[:, 8:14].T
        Z = [0.0] * m
        msgs = self.messages
        mu = msgs.get(u, {})

        def pick(vec):
            if vec is None:
                return Z
            if len(vec) != m:
                out = list(vec[:m])
                return out + [0.0] * (m - len(out))
            return vec

        X[rows["message_to_self"]] = np.array(pick(mu.get(u)))[:, None]
        X[rows["message_to_it"]] = np.array([pick(mu.get(v)) for v in candidates]).T
        X[rows["message_from_it"]] = np.array([pick(msgs.get(v, {}).get(u)) for v in candidates]).T
        X[rows["message_its_own"]] = np.array([pick(msgs.get(v, {}).get(v)) for v in candidates]).T
        X[rows["noise"]] = self.kind.draw_noise((n, cfg.random_input_amount), self.rng).T
        return X

    cls._precompute_features = _precompute_features
    cls._inputs = _inputs
