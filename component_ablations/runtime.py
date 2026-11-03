"""Strict resource loading and model construction (imported only by workers)."""
import copy
import hashlib
import os
from pathlib import Path
import numpy as np
import random
import torch

from rlkit.torch import pytorch_util as ptu
from rlkit.torch.algo.gentle import GENTLE
from .common import REPO, read_json, sha256, verify_file
from .step import MatchedStep


class ComponentGENTLE(MatchedStep, GENTLE):
    def __init__(self, *args, ablation_resources, project_root, masks, ablation_seed, **kwargs):
        self.ablation_resources = ablation_resources
        self.project_root = project_root
        self.mask_r, self.mask_v, self.mask_p = (int(masks[key]) for key in "RVP")
        self.ablation_seed = ablation_seed
        self.update_counts = dict(encoder=0, critic=0, actor=0, target_events=0, target_networks=0)
        self.real_batch_digest = hashlib.sha256()
        super().__init__(*args, **kwargs)
        if self.virtual_transition_use_cycle_weight or self.virtual_transition_use_recon_weight:
            raise ValueError("This loss contract permits decay-only weights")
        if not self.virtual_transition_train_policy_bc or not self.virtual_transition_train_policy_q:
            raise ValueError("User-defined P includes virtual critic, actor-Q AND actor-BC")
        if self.agent.use_ib:
            raise ValueError("Frozen source full configs are deterministic; IB is outside this experiment")

    def init_buffer(self, kwargs):
        entry = self.ablation_resources
        norms = np.load(str(verify_file(entry["normalizer"], self.project_root)), allow_pickle=False)
        for name in ("mean", "var", "count"):
            setattr(self.obs_normalizer, name, norms[name].copy())
        self.env.update_obs_mean_var(self.obs_normalizer.mean, self.obs_normalizer.var)
        observations = []
        task_counts = {task: 0 for task in list(self.train_tasks) + list(self.eval_tasks)}
        for record in entry["files"]:
            path = verify_file(record, self.project_root)
            array = np.load(str(path), allow_pickle=True)
            if len(array) != record["rows"]:
                raise ValueError("Dataset row count mismatch: " + str(path))
            task = record["task_id"]
            if record["split"] == "train":
                if task not in self.train_tasks:
                    raise ValueError("Training manifest includes held-out task")
                observations.extend(array[:, 0])
                buffer = self.train_buffer
            else:
                if task not in self.eval_tasks:
                    raise ValueError("Evaluation manifest includes training task")
                buffer = self.eval_buffer
            for i, row in enumerate(array):
                buffer.add_sample(task, self.obs_normalizer.forward(np.asarray(row[0], dtype=np.float64)),
                                  np.asarray(row[1], dtype=np.float64), row[2], int(i == len(array) - 1),
                                  self.obs_normalizer.forward(np.asarray(row[3], dtype=np.float64)), env_info={})
                task_counts[task] += 1
        if any(count == 0 for count in task_counts.values()):
            raise ValueError("Empty task in dataset manifest")
        # Verify the historical training-only RMS reconstruction; never replace
        # the frozen transform with these computed values or use eval rows.
        check = ptu.RunningMeanStd(shape=self.obs_dim)
        check.update(np.asarray(observations))
        for name in ("mean", "var", "count"):
            if not np.allclose(getattr(check, name), getattr(self.obs_normalizer, name), rtol=1e-9, atol=1e-9):
                raise ValueError("Training-data/normalizer provenance mismatch: " + name)

    def sample_sac(self, indices):
        batches = []
        for task in indices:
            buffer = self.train_buffer.task_buffers[task]
            rows = np.random.randint(0, buffer._size, self.batch_size)
            self.real_batch_digest.update(np.asarray([task], dtype=np.int64).tobytes())
            self.real_batch_digest.update(rows.astype(np.int64).tobytes())
            batches.append(self.unpack_batch(ptu.np_to_pytorch_batch(buffer.sample_data(rows))))
        return [torch.cat([batch[i] for batch in batches], 0) for i in range(5)]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_algorithm(config, resources, seed, root, cpu=False):
    from rlkit.envs import ENVS
    from rlkit.envs.wrappers import NormalizedBoxEnv
    from rlkit.torch.autoencoder import MlpEncoder, MlpDecoder
    from rlkit.torch.networks import FlattenMlp
    from rlkit.torch.sac.agent import Agent
    from rlkit.torch.sac.policies import TanhGaussianPolicy, ContextPolicyWrapper
    from rlkit.torch.multi_task_dynamics import MultiTaskDynamics

    if config["env_name"] not in ENVS:
        raise RuntimeError("Environment backend not registered; check per-environment MuJoCo paths")
    # The scheduler binds the physical GPU BEFORE importing torch. The native
    # helper rewrites CUDA_VISIBLE_DEVICES; avoid undoing that binding here.
    ptu.set_gpu_mode(False)
    if not cpu:
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA worker but CUDA is unavailable")
        ptu._use_gpu = True
        ptu.device = torch.device("cuda:0")
    env = NormalizedBoxEnv(ENVS[config["env_name"]](**config["env_params"]))
    reference_backend = resources["backend_reference"]
    version = reference_backend["engine_version"]
    engine = Path(os.environ["MUJOCO_PY_MJPRO_PATH"])
    library = engine / "bin" / ("libmujoco%d.so" % version)
    if config["env_name"] == "cheetah-vel":
        import gym
        xml = Path(gym.__file__).parent / "envs/mujoco/assets/half_cheetah.xml"
    else:
        xml = REPO / "rand_param_envs/gym/envs/mujoco/assets/walker2d.xml"
    for file_path, expected in ((library, reference_backend["engine_library_sha256"]), (xml, reference_backend["xml_sha256"])):
        if not file_path.is_file() or sha256(file_path) != expected:
            raise ValueError("Native MuJoCo library/XML differs from audited full backend: " + str(file_path))
    task_manifest = read_json(verify_file(resources["tasks"], root))
    native = env.wrapped_env
    original_tasks = copy.deepcopy(native.tasks)
    tasks = task_manifest["tasks"]
    if len(original_tasks) != len(tasks):
        raise ValueError("Environment task count mismatch")
    # Assert native backend task physics agrees before binding the fixed tasks.
    for generated, expected in zip(original_tasks, tasks):
        if generated.keys() != expected.keys():
            raise ValueError("Native task parameter keys differ from frozen manifest")
        for key in generated:
            if not np.allclose(np.asarray(generated[key]), np.asarray(expected[key]), rtol=1e-9, atol=1e-9):
                raise ValueError("Native backend task physics differs: " + key)
    native.tasks = [{key: np.asarray(value) if isinstance(value, list) else value for key, value in task.items()} for task in tasks]
    env.reset_task(0)
    seed_all(seed)
    env.seed(seed)
    obs_dim, action_dim = int(np.prod(env.observation_space.shape)), int(np.prod(env.action_space.shape))
    params = copy.deepcopy(config["algo_params"])
    latent, size = config["latent_size"], config["net_size"]
    next_obs = params["use_next_obs_in_context"]
    params["context_dim"] = obs_dim + action_dim + 1 + (obs_dim if next_obs else 0)
    encoder = MlpEncoder(hidden_sizes=[size] * 3, input_size=params["context_dim"], output_size=latent,
                         output_activation=torch.tanh, batch_attention=False)
    q1 = FlattenMlp(hidden_sizes=[size] * 3, input_size=obs_dim + action_dim + latent, output_size=1)
    q2 = FlattenMlp(hidden_sizes=[size] * 3, input_size=obs_dim + action_dim + latent, output_size=1)
    decoder = MlpDecoder(hidden_size=size, num_hidden_layers=3, z_dim=latent, action_dim=action_dim,
                         obs_dim=obs_dim, reward_dim=1, use_next_obs_in_context=next_obs)
    pre = resources["pretrained"][str(seed)]
    # Mandatory, strict loading; there is deliberately no random fallback.
    encoder.load(verify_file(pre["encoder"], root))
    decoder.load(verify_file(pre["decoder"], root))
    layers = 3 if next_obs else 2
    dynamics = MultiTaskDynamics(num_tasks=config["n_train_tasks"], hidden_size=size, num_hidden_layers=layers,
                                 action_dim=action_dim, obs_dim=obs_dim, reward_dim=1,
                                 use_next_obs_in_context=next_obs, ensemble_size=params["ensemble_size"],
                                 dynamics_weight_decay=[2.5e-5, 5e-5, 7.5e-5, 7.5e-5][:layers+1])
    for model, record in zip(dynamics.models, pre["dynamics"]):
        model.load_state_dict(torch.load(str(verify_file(record, root)), map_location="cpu", weights_only=True), strict=True)
        model.requires_grad_(False)
        model.eval()
    policy = TanhGaussianPolicy(hidden_sizes=[size] * 3, obs_dim=obs_dim + latent, latent_dim=latent, action_dim=action_dim)
    agent = Agent(latent, encoder, policy, **params)
    algorithm = ComponentGENTLE(env=env, train_tasks=task_manifest["train_tasks"], eval_tasks=task_manifest["eval_tasks"],
                                latent_dim=latent, nets=[[agent, ContextPolicyWrapper(policy, latent)], q1, q2, decoder, dynamics],
                                obs_normalizer=ptu.RunningMeanStd(shape=obs_dim), ablation_resources=resources,
                                project_root=root, masks=config["component_ablation"], ablation_seed=seed, **params)
    algorithm.to()
    algorithm.backend_identity = {"engine_version": version, "library": str(library), "library_sha256": sha256(library),
                                  "xml": str(xml), "xml_sha256": sha256(xml), "native_task_parameters_verified": True}
    return algorithm
