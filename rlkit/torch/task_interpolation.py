"""Offline task interpolation using supported, decoder-semantic graph paths.

This module deliberately needs neither environments nor task parameters.  The
graph and its narrow trust regions are *heuristics* for local task support, not
an estimator with a guarantee of recovering the true task manifold.  Pairwise
distances use shared inputs within each pair; they are empirical dissimilarities
and need not satisfy the triangle inequality.
"""

from contextlib import contextmanager
import heapq
import math

import torch


class SemanticTaskInterpolator:
    """Cache supported nonlinear edges, then sample along graph paths cheaply.

    ``refresh`` receives deterministic task embeddings and actual offline
    transitions, with shape [tasks, probes, feature].  The first half of each
    task's transitions supplies fitting probes and nearest-neighbor support;
    the second half supplies held-out validation and generated-transition
    inputs.  Use freshly sampled batches on each refresh.  Their independence
    is subject to the underlying replay sampler (duplicates are possible).

    ``decoder(obs, actions, z)`` must return reward, optionally concatenated
    with the ABSOLUTE next observation.  Parameters, training flags, and any
    existing parameter gradients are preserved across refresh.  No decoder
    calls or optimization are needed in ``sample``.
    """

    DEFAULTS = {
        'virtual_semantic_neighbors': 2,
        'virtual_semantic_probe_batch_size': 64,
        'virtual_semantic_refresh_interval': 100,
        'virtual_semantic_support_radius': 1.0,
        'virtual_semantic_min_shared_probes': 8,
        'virtual_semantic_max_latent_distance_ratio': 2.0,
        'virtual_semantic_max_reconstruction_error': 1.0,
        'virtual_semantic_reward_weight': 1.0,
        'virtual_semantic_dynamics_weight': 1.0,
        'virtual_semantic_path_nodes': 7,
        'virtual_semantic_path_steps': 8,
        'virtual_semantic_path_lr': 0.02,
        'virtual_semantic_trust_radius': 0.1,
        'virtual_semantic_latent_weight': 0.05,
        'virtual_semantic_validation_ratio': 1.25,
        'virtual_semantic_alpha_min': 0.1,
        'virtual_semantic_alpha_max': 0.9,
    }
    BATCH_KEYS = ('observations', 'actions', 'rewards',
                  'next_observations', 'terminals')

    @classmethod
    def validate_options(cls, options=None, **kwargs):
        supplied = dict(options or {})
        supplied.update(kwargs)
        unknown = set(supplied) - set(cls.DEFAULTS)
        if unknown:
            raise ValueError('Unknown semantic interpolation options: {}'.format(
                ', '.join(sorted(unknown))))
        result = dict(cls.DEFAULTS)
        result.update(supplied)
        integer_names = ('neighbors', 'probe_batch_size', 'refresh_interval',
                         'min_shared_probes', 'path_nodes', 'path_steps')
        for name in integer_names:
            key = 'virtual_semantic_' + name
            value = result[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError('{} must be an integer'.format(key))
            if not math.isfinite(value) or int(value) != value:
                raise ValueError('{} must be an integer'.format(key))
            result[key] = int(value)
            minimum = 0 if name == 'path_steps' else (3 if name == 'path_nodes' else 1)
            if result[key] < minimum:
                raise ValueError('{} must be >= {}'.format(key, minimum))
        for key in set(result) - {'virtual_semantic_' + n for n in integer_names}:
            value = result[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError('{} must be a finite number'.format(key))
            result[key] = float(value)
            if not math.isfinite(result[key]):
                raise ValueError('{} must be a finite number'.format(key))
        for name in ('support_radius', 'max_latent_distance_ratio',
                     'max_reconstruction_error', 'path_lr'):
            if result['virtual_semantic_' + name] <= 0:
                raise ValueError('virtual_semantic_{} must be positive'.format(name))
        for name in ('reward_weight', 'dynamics_weight', 'trust_radius', 'latent_weight'):
            if result['virtual_semantic_' + name] < 0:
                raise ValueError('virtual_semantic_{} must be nonnegative'.format(name))
        if result['virtual_semantic_validation_ratio'] < 1:
            raise ValueError('virtual_semantic_validation_ratio must be >= 1')
        if not (0 <= result['virtual_semantic_alpha_min'] <
                result['virtual_semantic_alpha_max'] <= 1):
            raise ValueError('semantic alpha bounds must satisfy 0 <= min < max <= 1')
        if (result['virtual_semantic_min_shared_probes'] >
                result['virtual_semantic_probe_batch_size']):
            raise ValueError('semantic min_shared_probes cannot exceed probe_batch_size')
        return result

    def __init__(self, decoder, use_next_obs_in_context, **options):
        self.decoder = decoder
        self.use_next_obs_in_context = bool(use_next_obs_in_context)
        self.options = self.validate_options(options)
        if self._opt('reward_weight') == 0 and (
                not self.use_next_obs_in_context or self._opt('dynamics_weight') == 0):
            raise ValueError('At least one available decoder output needs a positive semantic weight')
        self.edges = {}
        self.paths = []
        self.stats = {}
        self._refresh_count = 0
        self.task_z = None

    def _opt(self, name):
        return self.options['virtual_semantic_' + name]

    def _count(self, name, amount=1):
        key = 'virtual_semantic_' + name
        self.stats[key] = self.stats.get(key, 0) + amount

    @contextmanager
    def _fixed_decoder(self):
        parameters = list(self.decoder.parameters())
        requires_grad = [p.requires_grad for p in parameters]
        modes = [(module, module.training) for module in self.decoder.modules()]
        try:
            for parameter in parameters:
                parameter.requires_grad_(False)
            self.decoder.eval()
            yield
        finally:
            for parameter, required in zip(parameters, requires_grad):
                parameter.requires_grad_(required)
            # Preserve individually configured submodule modes as well.
            for module, mode in modes:
                module.training = mode

    def _features(self, z, observations, actions):
        """Predictions on exactly the same input bank for every supplied z."""
        n, p = z.shape[0], observations.shape[0]
        obs = observations.unsqueeze(0).expand(n, p, -1)
        act = actions.unsqueeze(0).expand(n, p, -1)
        repeated_z = z.unsqueeze(1).expand(n, p, -1)
        prediction = self.decoder(obs, act, repeated_z)
        return self._scale_prediction(prediction, obs)

    def _scale_prediction(self, prediction, observations):
        reward = prediction[..., :1] / self.reward_scale
        pieces = [reward * math.sqrt(self._opt('reward_weight'))]
        if self.use_next_obs_in_context:
            delta = (prediction[..., 1:] - observations) / self.delta_scale
            pieces.append(delta * math.sqrt(self._opt('dynamics_weight') / delta.shape[-1]))
        return torch.cat(pieces, dim=-1)

    @staticmethod
    def _energy(features):
        # Discrete integral of squared speed on a unit-time path.  The feature
        # axis is summed, and probes averaged, so state dimensionality is
        # controlled by the explicit dynamics normalization above.
        increments = features[1:] - features[:-1]
        return increments.square().sum(-1).mean(-1).sum() * (features.shape[0] - 1)

    @staticmethod
    def _densify(nodes):
        result = torch.empty((2 * nodes.shape[0] - 1, nodes.shape[1]),
                             dtype=nodes.dtype, device=nodes.device)
        result[::2] = nodes
        result[1::2] = (nodes[1:] + nodes[:-1]) * 0.5
        return result

    def _reset_stats(self):
        names = ('candidate_edges', 'supported_edges', 'edges', 'paths',
                 'rejected_support', 'rejected_reconstruction', 'rejected_latent',
                 'rejected_nonfinite', 'rejected_degenerate', 'rejected_validation',
                 'refinement_attempts', 'refinement_accepted', 'refinement_rejected',
                 'reference_fallbacks', 'sampled', 'rejected_no_path',
                 'refinement_displacement_mean', 'heldout_reconstruction_mean',
                 'path_hops_mean', 'sampled_quality_mean', 'covered_tasks',
                 'task_coverage_fraction', 'supported_probe_count_mean',
                 'validation_energy_ratio_mean', 'sampled_unique_edges',
                 'sampled_support_unique_fraction')
        self.stats = {'virtual_semantic_' + name: 0 for name in names}
        self.stats['virtual_semantic_refresh_count'] = self._refresh_count

    def _validate_batches(self, task_z, batches):
        if task_z.ndim != 2 or not task_z.is_floating_point():
            raise ValueError('task_z must be a floating [tasks, latent_dim] tensor')
        missing = set(self.BATCH_KEYS) - set(batches)
        if missing:
            raise ValueError('Missing semantic probe arrays: {}'.format(sorted(missing)))
        shape = batches['observations'].shape[:2]
        if len(shape) != 2 or shape[0] != task_z.shape[0] or shape[1] < 2:
            raise ValueError('Probe arrays must have [tasks, >=2 probes, features] shape')
        for key in self.BATCH_KEYS:
            value = batches[key]
            if value.ndim != 3 or value.shape[:2] != shape:
                raise ValueError('Incompatible semantic probe shape for {}'.format(key))
            if value.device != task_z.device:
                raise ValueError('Semantic embeddings and probe arrays must share a device')
        if batches['rewards'].shape[-1] != 1 or batches['terminals'].shape[-1] != 1:
            raise ValueError('Semantic rewards and terminals must have one feature')
        if batches['next_observations'].shape != batches['observations'].shape:
            raise ValueError('Semantic observations and next observations must match')

    @torch.no_grad()
    def refresh(self, task_z, batches):
        """Rebuild graph from the CURRENT encoder/decoder and offline data."""
        self._validate_batches(task_z, batches)
        self._refresh_count += 1
        self._reset_stats()
        self.edges, self.paths = {}, []
        self.task_z = task_z.detach().clone()
        n_tasks = task_z.shape[0]
        if n_tasks < 2:
            self._count('rejected_no_path')
            return
        if not torch.isfinite(task_z).all() or any(
                not torch.isfinite(batches[key]).all() for key in self.BATCH_KEYS):
            self._count('rejected_nonfinite')
            return
        half = batches['observations'].shape[1] // 2
        fit_count = min(half, self._opt('probe_batch_size'))
        check_count = min(batches['observations'].shape[1] - half,
                          self._opt('probe_batch_size'))
        fit = {key: batches[key][:, :fit_count].detach() for key in self.BATCH_KEYS}
        check = {key: batches[key][:, half:half + check_count].detach()
                 for key in self.BATCH_KEYS}
        # Sampling later must remain valid even if caller reuses a batch buffer.
        self.check = {key: value.reshape(-1, value.shape[-1]).clone()
                      for key, value in check.items()}
        self.check_sources = torch.arange(n_tasks, device=task_z.device).repeat_interleave(check_count)
        self.reward_scale = fit['rewards'].reshape(-1, 1).std(0, unbiased=False).clamp_min(1e-3)
        delta = fit['next_observations'] - fit['observations']
        self.delta_scale = delta.reshape(-1, delta.shape[-1]).std(0, unbiased=False).clamp_min(1e-3)
        fit_sa = torch.cat([fit['observations'], fit['actions']], dim=-1)
        check_sa = torch.cat([check['observations'], check['actions']], dim=-1)
        flat_fit_sa = fit_sa.reshape(-1, fit_sa.shape[-1])
        center = flat_fit_sa.mean(0)
        scale = flat_fit_sa.std(0, unbiased=False).clamp_min(1e-3)
        fit_sa = (fit_sa - center) / scale
        check_sa = (check_sa - center) / scale
        flat_fit_sa = fit_sa.reshape(-1, fit_sa.shape[-1])
        flat_check_sa = check_sa.reshape(-1, check_sa.shape[-1])
        # RMS nearest-neighbor distance, after per-coordinate standardization.
        normalizer = math.sqrt(fit_sa.shape[-1])
        fit_support = torch.stack([
            torch.cdist(flat_fit_sa, fit_sa[i]).min(-1).values / normalizer
            for i in range(n_tasks)])
        check_support = torch.stack([
            torch.cdist(flat_check_sa, fit_sa[i]).min(-1).values / normalizer
            for i in range(n_tasks)])
        self._flat_fit = {key: value.reshape(-1, value.shape[-1])
                          for key, value in fit.items()}
        with self._fixed_decoder():
            predicted = self.decoder(
                check['observations'], check['actions'],
                task_z[:, None, :].expand(-1, check_count, -1))
            actual = check['rewards']
            if self.use_next_obs_in_context:
                actual = torch.cat([actual, check['next_observations']], dim=-1)
            residual = (self._scale_prediction(predicted, check['observations']) -
                        self._scale_prediction(actual, check['observations']))
            reconstruction_error = residual.square().sum(-1).mean(-1).sqrt()
            finite_errors = reconstruction_error[torch.isfinite(reconstruction_error)]
            if finite_errors.numel():
                self.stats['virtual_semantic_heldout_reconstruction_mean'] = finite_errors.mean().item()
            reliable = (torch.isfinite(reconstruction_error) &
                        (reconstruction_error <= self._opt('max_reconstruction_error')))
            all_features = self._features(task_z, self._flat_fit['observations'],
                                          self._flat_fit['actions'])
            distances = task_z.new_full((n_tasks, n_tasks), float('inf'))
            latent_distance = torch.cdist(task_z, task_z)
            positive_latent = latent_distance.masked_fill(latent_distance <= 1e-8, float('inf'))
            local_scale = positive_latent.min(-1).values
            candidates = {}
            for i in range(n_tasks):
                for j in range(i + 1, n_tasks):
                    self._count('candidate_edges')
                    if not (reliable[i] and reliable[j]):
                        self._count('rejected_reconstruction')
                        continue
                    length = latent_distance[i, j]
                    if length <= 1e-8:
                        self._count('rejected_degenerate')
                        continue
                    if length > self._opt('max_latent_distance_ratio') * torch.minimum(local_scale[i], local_scale[j]):
                        self._count('rejected_latent')
                        continue
                    fit_ids = self._shared_indices(i, j, fit_count, fit_support)
                    check_ids = self._shared_indices(i, j, check_count, check_support)
                    if min(fit_ids.numel(), check_ids.numel()) < self._opt('min_shared_probes'):
                        self._count('rejected_support')
                        continue
                    fit_ids = self._subsample(fit_ids)
                    check_ids = self._subsample(check_ids)
                    difference = all_features[i, fit_ids] - all_features[j, fit_ids]
                    semantic_distance = difference.square().sum(-1).mean().sqrt()
                    if not torch.isfinite(semantic_distance):
                        self._count('rejected_nonfinite')
                        continue
                    if semantic_distance <= 1e-8:
                        self._count('rejected_degenerate')
                        continue
                    distances[i, j] = distances[j, i] = semantic_distance
                    candidates[(i, j)] = (fit_ids, check_ids)
                    self._count('supported_edges')
            neighbors = torch.zeros_like(distances, dtype=torch.bool)
            k = min(self._opt('neighbors'), n_tasks - 1)
            values, indices = distances.topk(k, dim=-1, largest=False)
            neighbors.scatter_(1, indices, torch.isfinite(values))
            displacements = []
            for (i, j), (fit_ids, check_ids) in candidates.items():
                if not (neighbors[i, j] and neighbors[j, i]):
                    continue
                edge = self._build_edge(i, j, fit_ids, check_ids)
                if edge is None:
                    continue
                # This is a discount from observable real endpoint fit, NOT a
                # calibrated confidence probability at a virtual task.
                error = torch.maximum(reconstruction_error[i], reconstruction_error[j])
                edge['quality'] = torch.exp(-error.square()).detach()
                self.edges[(i, j)] = edge
                displacements.append(edge['displacement'])
            if displacements:
                self.stats['virtual_semantic_refinement_displacement_mean'] = sum(displacements) / len(displacements)
        self.stats['virtual_semantic_edges'] = len(self.edges)
        covered = {task for endpoints in self.edges for task in endpoints}
        self.stats['virtual_semantic_covered_tasks'] = len(covered)
        self.stats['virtual_semantic_task_coverage_fraction'] = len(covered) / n_tasks
        if self.edges:
            self.stats['virtual_semantic_supported_probe_count_mean'] = sum(
                edge['check_ids'].numel() for edge in self.edges.values()) / len(self.edges)
            self.stats['virtual_semantic_validation_energy_ratio_mean'] = sum(
                edge['validation_energy_ratio'] for edge in self.edges.values()) / len(self.edges)
        self._build_graph_paths(n_tasks)
        self.stats['virtual_semantic_paths'] = len(self.paths)
        if self.paths:
            self.stats['virtual_semantic_path_hops_mean'] = sum(
                len(path['edges']) for path in self.paths) / len(self.paths)
        else:
            self._count('rejected_no_path')

    def _shared_indices(self, i, j, count, support):
        ids = torch.cat([torch.arange(i * count, (i + 1) * count, device=support.device),
                         torch.arange(j * count, (j + 1) * count, device=support.device)])
        mask = ((support[i, ids] <= self._opt('support_radius')) &
                (support[j, ids] <= self._opt('support_radius')))
        return ids[mask]

    def _subsample(self, indices):
        if indices.numel() > self._opt('probe_batch_size'):
            indices = indices[torch.randperm(indices.numel(), device=indices.device)[:self._opt('probe_batch_size')]]
        return indices

    def _project_offsets(self, offsets):
        with torch.no_grad():
            norms = offsets.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            offsets.mul_((self._opt('trust_radius') / norms).clamp(max=1.0))
            padded = torch.cat([torch.zeros_like(offsets[:1]), offsets,
                                torch.zeros_like(offsets[:1])], dim=0)
            max_jump = (padded[1:] - padded[:-1]).norm(dim=-1).max()
            # Triangle inequality then bounds every final latent segment by
            # twice the length of a segment in the straight reference edge.
            cap = 1.0 / (self._opt('path_nodes') - 1)
            offsets.mul_((cap / max_jump.clamp_min(1e-12)).clamp(max=1.0))

    def _refine(self, reference, obs, actions, fit_energy):
        if self._opt('path_steps') == 0 or self._opt('trust_radius') == 0:
            return reference, False
        self._count('refinement_attempts')
        length = (reference[-1] - reference[0]).norm().detach()
        with torch.enable_grad():
            # Dimensionless offsets make learning rate independent of the
            # absolute scale of the learned latent representation.
            offsets = torch.zeros_like(reference[1:-1], requires_grad=True)
            optimizer = torch.optim.Adam([offsets], lr=self._opt('path_lr'))
            best = reference.clone()
            best_loss = float('inf')
            for step in range(self._opt('path_steps') + 1):
                nodes = torch.cat([reference[:1], reference[1:-1] + length * offsets,
                                   reference[-1:]], dim=0)
                features = self._features(nodes, obs, actions)
                latent_energy = (nodes[1:] - nodes[:-1]).square().sum() * (nodes.shape[0] - 1)
                loss = (self._energy(features) / fit_energy.clamp_min(1e-12) +
                        self._opt('latent_weight') * latent_energy / length.square().clamp_min(1e-12))
                if not torch.isfinite(loss):
                    return reference, True
                if loss.item() < best_loss:
                    best_loss, best = loss.item(), nodes.detach().clone()
                if step == self._opt('path_steps'):
                    break
                gradient, = torch.autograd.grad(loss, offsets, allow_unused=False)
                if not torch.isfinite(gradient).all():
                    return reference, True
                optimizer.zero_grad(set_to_none=True)
                offsets.grad = gradient
                optimizer.step()
                self._project_offsets(offsets)
        return best, False

    def _build_edge(self, i, j, fit_ids, check_ids):
        t = torch.linspace(0, 1, self._opt('path_nodes'),
                           dtype=self.task_z.dtype, device=self.task_z.device)[:, None]
        reference = (1 - t) * self.task_z[i] + t * self.task_z[j]
        fit_obs, fit_actions = (self._flat_fit[key][fit_ids] for key in ('observations', 'actions'))
        check_obs, check_actions = (self.check[key][check_ids] for key in ('observations', 'actions'))
        dense_reference = self._densify(reference)
        fit_features = self._features(reference, fit_obs, fit_actions)
        reference_check = self._features(dense_reference, check_obs, check_actions)
        if not (torch.isfinite(fit_features).all() and torch.isfinite(reference_check).all()):
            self._count('rejected_nonfinite')
            return None
        baseline_check_energy = self._energy(reference_check)
        if baseline_check_energy <= 1e-12:
            self._count('rejected_degenerate')
            return None
        nodes, failed = self._refine(reference, fit_obs, fit_actions, self._energy(fit_features))
        length = (reference[-1] - reference[0]).norm()
        dense_nodes = self._densify(nodes)
        check_features = self._features(dense_nodes, check_obs, check_actions)
        check_energy = self._energy(check_features)
        valid = (torch.isfinite(nodes).all() and torch.isfinite(check_features).all() and
                 torch.isfinite(check_energy) and
                 check_energy <= self._opt('validation_ratio') * baseline_check_energy + 1e-10 and
                 (dense_nodes - dense_reference).norm(dim=-1).max() <=
                 self._opt('trust_radius') * length + 1e-6 and
                 (nodes[1:] - nodes[:-1]).norm(dim=-1).max() <=
                 2 * length / (nodes.shape[0] - 1) + 1e-6 and
                 torch.equal(nodes[0], reference[0]) and torch.equal(nodes[-1], reference[-1]))
        if failed or not valid:
            self._count('refinement_rejected')
            self._count('reference_fallbacks')
            if not valid:
                self._count('rejected_validation')
            # This fallback has already passed endpoint reconstruction,
            # shared-input support, and finite held-out midpoint checks.
            nodes, dense_nodes, check_features = reference, dense_reference, reference_check
        displacement = (nodes - reference).norm(dim=-1).max().item()
        if displacement > 1e-8:
            self._count('refinement_accepted')
        # Arc length is measured on held-out inputs, including the internal
        # midpoint checks.  Store these dense nodes as the actual sampling path.
        increments = check_features[1:] - check_features[:-1]
        segment_lengths = increments.square().sum(-1).mean(-1).sqrt()
        if not torch.isfinite(segment_lengths).all() or segment_lengths.sum() <= 1e-8:
            self._count('rejected_degenerate')
            return None
        return {'nodes': dense_nodes.detach(), 'lengths': segment_lengths.detach(),
                'length': segment_lengths.sum().item(), 'check_ids': check_ids,
                'validation_energy_ratio': (self._energy(check_features) / baseline_check_energy).item(),
                'displacement': displacement, 'endpoints': (i, j)}

    def _build_graph_paths(self, n_tasks):
        adjacency = [[] for _ in range(n_tasks)]
        for (i, j), edge in self.edges.items():
            adjacency[i].append((j, edge['length']))
            adjacency[j].append((i, edge['length']))
        for source in range(n_tasks):
            distance, previous = {source: 0.0}, {}
            queue = [(0.0, source)]
            while queue:
                current_distance, current = heapq.heappop(queue)
                if current_distance > distance[current]:
                    continue
                for neighbor, weight in adjacency[current]:
                    proposal = current_distance + weight
                    if proposal < distance.get(neighbor, float('inf')):
                        distance[neighbor], previous[neighbor] = proposal, current
                        heapq.heappush(queue, (proposal, neighbor))
            for target in range(source + 1, n_tasks):
                if target not in previous:
                    continue
                pairs, node = [], target
                while node != source:
                    parent = previous[node]
                    pairs.append((parent, node))
                    node = parent
                pairs.reverse()
                lengths = []
                for start, end in pairs:
                    lengths.append(self.edges[tuple(sorted((start, end)))]['length'])
                self.paths.append({'edges': pairs, 'lengths': lengths,
                                   'length': sum(lengths)})

    @torch.no_grad()
    def sample(self, num_tasks, batch_size):
        """Return embeddings AND their supported real transition inputs.

        A result may use both endpoint tasks' data; ``anchor_positions`` is
        descriptive metadata only.  Never use it to resample arbitrary inputs.
        """
        for name in ('sampled_quality_mean', 'sampled_unique_edges', 'sampled_support_unique_fraction'):
            self.stats['virtual_semantic_' + name] = 0
        if num_tasks <= 0 or batch_size <= 0:
            return None
        if not self.paths:
            self._count('rejected_no_path')
            return None
        results = {key: [] for key in ('embeddings', 'observations', 'actions',
                                      'next_observations', 'terminals',
                                      'quality_weights', 'anchor_positions')}
        device = self.task_z.device
        sampled_edges, unique_fraction_sum = set(), 0.0
        for _ in range(num_tasks):
            path = self.paths[torch.randint(len(self.paths), (), device=device).item()]
            alpha = (self._opt('alpha_min') + torch.rand((), device=device).item() *
                     (self._opt('alpha_max') - self._opt('alpha_min')))
            # Cached paths are stored once, from the lower task index. Random
            # orientation keeps conservative alpha ranges near BOTH endpoints
            # instead of systematically favoring low-index training tasks.
            if torch.rand((), device=device).item() < 0.5:
                alpha = 1.0 - alpha
            remaining = alpha * path['length']
            edge_position = len(path['edges']) - 1
            for position, edge_length in enumerate(path['lengths']):
                if remaining < edge_length or position == len(path['edges']) - 1:
                    edge_position = position
                    break
                remaining -= edge_length
            start, end = path['edges'][edge_position]
            edge = self.edges[tuple(sorted((start, end)))]
            sampled_edges.add(tuple(sorted((start, end))))
            forward = start < end
            nodes = edge['nodes'] if forward else edge['nodes'].flip(0)
            lengths = edge['lengths'] if forward else edge['lengths'].flip(0)
            cumulative = lengths.cumsum(0)
            location = lengths.new_tensor(remaining).clamp(0, cumulative[-1])
            segment = min(torch.searchsorted(cumulative, location, right=True).item(),
                          lengths.numel() - 1)
            before = cumulative[segment - 1] if segment else 0.0
            fraction = ((location - before) / lengths[segment].clamp_min(1e-12)).clamp(0, 1)
            embedding = (1 - fraction) * nodes[segment] + fraction * nodes[segment + 1]
            ids = edge['check_ids'][torch.randint(edge['check_ids'].numel(),
                                                 (batch_size,), device=device)]
            unique_fraction_sum += ids.unique().numel() / batch_size
            results['embeddings'].append(embedding)
            for key in ('observations', 'actions', 'next_observations', 'terminals'):
                results[key].append(self.check[key][ids])
            results['quality_weights'].append(edge['quality'].reshape(1))
            anchor = start if segment + fraction.item() < (nodes.shape[0] - 1) / 2 else end
            results['anchor_positions'].append(torch.tensor(anchor, device=device, dtype=torch.long))
        self._count('sampled', num_tasks)
        result = {key: torch.stack(values).detach() for key, values in results.items()}
        self.stats['virtual_semantic_sampled_quality_mean'] = result['quality_weights'].mean().item()
        self.stats['virtual_semantic_sampled_unique_edges'] = len(sampled_edges)
        self.stats['virtual_semantic_sampled_support_unique_fraction'] = unique_fraction_sum / num_tasks
        return result
