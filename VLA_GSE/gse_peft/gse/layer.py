"""
GSE (Generalized and Specialized Expert) Layer implementations
"""
import math
import logging
from abc import ABC
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.tuners.tuners_utils import BaseTunerLayer

logger = logging.getLogger(__name__)


class TopKGSELayer(nn.Module):
    """
    Top-K gated MoE layer with Generalized and Specialized Experts.
    
    Generalized experts are always selected (no routing).
    Specialized experts are selected by top-k gating.
    """
    
    def __init__(
        self, 
        generalized_experts: nn.ModuleList, 
        specialized_experts: nn.ModuleList, 
        gate: nn.Module, 
        top_k: int,
        compute_aux_loss: bool = True,
    ):
        super().__init__()
        self.generalized_experts = generalized_experts  # Always activated
        self.specialized_experts = specialized_experts  # Selected by router
        self.gate = gate  # Only routes specialized experts
        self.top_k = top_k
        self.compute_aux_loss = compute_aux_loss
        self.layer_loss = None
        self.expert_sum = torch.zeros((1, len(self.specialized_experts)))
        self.expert_sum_now = torch.zeros((1, len(self.specialized_experts)))
        self.aux_tot = 0
        self.merge_tot = 0
    
    def get_expert_similarity_loss(self):
        """Compute similarity loss between specialized experts"""
        if len(self.specialized_experts) == 0:
            return torch.tensor(0.0)
        lora_A_flatten = [torch.flatten(expert.lora_A.weight).view(1, -1) for expert in self.specialized_experts]
        lora_A_flatten = torch.cat(lora_A_flatten, dim=0)
        norm_A = torch.norm(lora_A_flatten, dim=1, keepdim=True)
        lora_A_score = lora_A_flatten @ lora_A_flatten.T / (norm_A @ norm_A.T + 1e-8)
        sim_loss = lora_A_score.fill_diagonal_(0).sum()
        return sim_loss
    
    def get_layer_loss(self, gate_logits: torch.Tensor, selected_experts: torch.Tensor) -> torch.Tensor:
        """Compute load balancing loss for specialized experts"""
        if len(self.specialized_experts) == 0:
            return torch.tensor(0.0, device=gate_logits.device)
        num_inputs = gate_logits.shape[0]
        num_experts = len(self.specialized_experts)
        expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=num_experts)
        expert_fractions = expert_counts / num_inputs
        expert_probs = torch.sum(gate_logits, dim=0) / num_inputs
        layer_loss = num_experts * torch.sum(expert_fractions * expert_probs)
        return layer_loss
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        flattened_inputs = inputs.view((-1, inputs.shape[-1]))
        
        # Initialize results
        if len(self.generalized_experts) > 0:
            results = torch.zeros_like(self.generalized_experts[0](flattened_inputs))
        elif len(self.specialized_experts) > 0:
            results = torch.zeros_like(self.specialized_experts[0](flattened_inputs))
        else:
            return torch.zeros((*inputs.shape[:-1], inputs.shape[-1]), device=inputs.device, dtype=inputs.dtype)
        
        # Generalized experts: always activated with equal weight
        num_gen = len(self.generalized_experts)
        if num_gen > 0:
            gen_weight = 1.0 / num_gen  # Equal weight for generalized experts
            for expert in self.generalized_experts:
                results += gen_weight * expert(flattened_inputs)
        
        # Specialized experts: selected by top-k router
        num_spec = len(self.specialized_experts)
        if num_spec > 0 and self.top_k > 0:
            gate_logits = F.softmax(self.gate(flattened_inputs), dim=-1)
            weights, selected_experts = torch.topk(input=gate_logits, k=min(self.top_k, num_spec), dim=-1)
            weights = weights / torch.sum(weights, dim=-1, keepdim=True, dtype=inputs.dtype)
            
            for i, expert in enumerate(self.specialized_experts):
                batch_idx, nth_expert = torch.where(selected_experts == i)
                results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(flattened_inputs[batch_idx])
            
            if self.compute_aux_loss:
                self.layer_loss = self.get_layer_loss(gate_logits=gate_logits, selected_experts=selected_experts)
            else:
                self.layer_loss = None
        else:
            self.layer_loss = None
        
        results = results.view((*inputs.shape[:-1], results.shape[-1]))
        return results


class GSEExpert(nn.Module):
    """Single expert in GSE MoE"""
    
    def __init__(self, lora_A: nn.Module, lora_B: nn.Module, lora_dropout: nn.Module, scaling: float):
        super().__init__()
        self.lora_A = lora_A
        self.lora_B = lora_B
        self.lora_dropout = lora_dropout
        self.register_buffer("scaling", torch.tensor(float(scaling), dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.lora_B(self.lora_A(self.lora_dropout(inputs))) * self.scaling
        return outputs


class GSELayer(BaseTunerLayer, ABC):
    """Base GSE Layer with Generalized and Specialized Experts"""
    
    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.lora_rank = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        self.kwargs = kwargs
        
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features
        
        self.lora_gating = nn.ModuleDict({})
        self.moe_layer = nn.ModuleDict({})

    def update_layer(
        self, adapter_name: str, lora_rank: int, lora_alpha: int, lora_dropout: float, init_lora_weights: bool,
        num_experts: int, num_generalized_experts: int, top_k: int, init_type: str, init_cof: float = None,
        scaling_factor: int = None, specialized_scaling_method: str = "default",
        specialized_scaling_base: float = 2.0, specialized_scaling_eps: float = 1e-12,
        skip_svd_init: bool = False, aux_loss_weight: float = 0.01,
    ) -> None:
        """Update the GSE layer with new adapter"""
        if lora_rank <= 0:
            raise ValueError(f"The rank `r` should be a positive integer value but the value passed is {lora_rank}.")

        num_specialized_experts = num_experts - num_generalized_experts
        
        # Distribute rank among experts
        # Generalized experts get larger ranks (from largest singular values)
        # Specialized experts get remaining ranks
        base_rank = lora_rank // num_experts
        extra_rank = lora_rank % num_experts
        
        # Generalized experts get slightly larger ranks
        gen_rank_list = [base_rank + (1 if i < extra_rank and i < num_generalized_experts else 0) 
                        for i in range(num_generalized_experts)]
        spec_rank_list = [base_rank + (1 if i + num_generalized_experts < extra_rank else 0) 
                         for i in range(num_specialized_experts)]
        
        rank_list = gen_rank_list + spec_rank_list
        assert 0 not in rank_list, f"Rank per expert is 0. Increase lora_rank or decrease num_experts."
        
        self.lora_rank[adapter_name] = lora_rank
        self.lora_alpha[adapter_name] = lora_alpha

        if lora_dropout > 0.0:
            self.lora_dropout[adapter_name] = nn.ModuleList(nn.Dropout(p=lora_dropout) for _ in range(num_experts))
        else:
            self.lora_dropout[adapter_name] = nn.ModuleList(nn.Identity() for _ in range(num_experts))
            
        self.lora_A[adapter_name] = nn.ModuleList(
            nn.Linear(self.in_features, rank_list[i], bias=False) for i in range(num_experts))
        self.lora_B[adapter_name] = nn.ModuleList(
            nn.Linear(rank_list[i], self.out_features, bias=False) for i in range(num_experts))
        
        need_svd = "gse" in init_type or "goat" in init_type or "mgse" in init_type or "svd" in init_type
        rho = float(os.getenv("RHO", 10))
        eta = float(os.getenv("ETA", 1.0))
        
        if "gse" in init_type or "goat" in init_type or "mgse" in init_type:
            self.scaling[adapter_name] = [math.sqrt(3 * eta * self.in_features / rank_list[i]) for i in range(num_experts)]
        else:
            self.scaling[adapter_name] = [lora_alpha / rank_list[i] for i in range(num_experts)]

        specialized_scaling_method = str(specialized_scaling_method).lower()
        gradient_balanced_scaling = specialized_scaling_method in {
            "gradient_scale_balancing", "gsb", "trace_inverse"
        }
        if specialized_scaling_method != "default" and not gradient_balanced_scaling:
            raise ValueError(
                "specialized_scaling_method must be 'default' or "
                "'gradient_scale_balancing'/'gsb'/'trace_inverse', "
                f"got {specialized_scaling_method!r}"
            )
        
        # Gate only for specialized experts
        if num_specialized_experts > 0:
            self.lora_gating[adapter_name] = nn.Linear(self.in_features, num_specialized_experts, bias=False)
        else:
            self.lora_gating[adapter_name] = None
            
        if init_type == "lora" or skip_svd_init:
            for i in range(len(self.lora_A[adapter_name])):
                nn.init.kaiming_uniform_(self.lora_A[adapter_name][i].weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B[adapter_name][i].weight)
        elif need_svd:
            def svd_init():
                weight = self.get_base_layer().weight
                dtype = weight.dtype
                weight = weight.to(torch.float32)
                orig_device = weight.device

                if orig_device.type == "cpu" and torch.cuda.is_available():
                    weight = weight.cuda()

                t0 = time.perf_counter()

                if "mgse" in init_type:
                    U_full, S_full, Vh_full = torch.linalg.svd(
                        weight.data, full_matrices=False)
                    if weight.device.type != orig_device.type:
                        U_full = U_full.to(orig_device)
                        S_full = S_full.to(orig_device)
                        Vh_full = Vh_full.to(orig_device)
                    Vr = U_full[:, -lora_rank:]
                    S = S_full[-lora_rank:]
                    Uhr = Vh_full[-lora_rank:, :]
                else:
                    niter = int(os.getenv("GSE_SVD_NITER", 2))
                    svd_seed = int(os.getenv("GSE_SVD_SEED", 42))
                    rng_state = torch.random.get_rng_state()
                    cuda_rng = None
                    if weight.is_cuda:
                        cuda_rng = torch.cuda.get_rng_state(weight.device)
                    torch.manual_seed(svd_seed)
                    if weight.is_cuda:
                        torch.cuda.manual_seed(svd_seed)
                    try:
                        U_lr, S_lr, Vright_lr = torch.svd_lowrank(
                            weight.data, q=lora_rank, niter=niter)
                        Vr = U_lr
                        S = S_lr
                        Uhr = Vright_lr.T
                    except Exception:
                        logger.warning(
                            "svd_lowrank failed, falling back to full SVD")
                        U_full, S_full, Vh_full = torch.linalg.svd(
                            weight.data, full_matrices=False)
                        Vr = U_full[:, :lora_rank]
                        S = S_full[:lora_rank]
                        Uhr = Vh_full[:lora_rank, :]
                    finally:
                        torch.random.set_rng_state(rng_state)
                        if cuda_rng is not None:
                            torch.cuda.set_rng_state(cuda_rng, weight.device)

                    if weight.device.type != orig_device.type:
                        Vr = Vr.to(orig_device)
                        S = S.to(orig_device)
                        Uhr = Uhr.to(orig_device)

                elapsed = time.perf_counter() - t0
                logger.debug(
                    f"SVD init ({init_type}) for "
                    f"{weight.shape[0]}x{weight.shape[1]}: {elapsed:.3f}s")

                if weight.device.type != orig_device.type:
                    weight.data = weight.data.to(orig_device)

                expert_ranges = []
                sum_rank = 0
                for ri in rank_list:
                    expert_ranges.append((sum_rank, sum_rank + ri))
                    sum_rank += ri

                if gradient_balanced_scaling and num_specialized_experts > 0:
                    spec_traces = torch.stack([
                        S[start:end].sum()
                        for start, end in expert_ranges[num_generalized_experts:]
                    ])
                    trace_mean = spec_traces.mean()
                    for offset, trace in enumerate(spec_traces):
                        expert_idx = num_generalized_experts + offset
                        balanced_scale = (
                            float(specialized_scaling_base)
                            * trace_mean
                            / trace.clamp_min(float(specialized_scaling_eps))
                        )
                        self.scaling[adapter_name][expert_idx] = float(balanced_scale.item())
                    logger.debug(
                        "Applied Gradient Scale Balancing to %d specialized experts",
                        num_specialized_experts,
                    )

                if gradient_balanced_scaling:
                    lora_A = torch.empty(
                        (lora_rank, Uhr.shape[1]), device=Uhr.device, dtype=Uhr.dtype)
                    lora_B = torch.empty(
                        (Vr.shape[0], lora_rank), device=Vr.device, dtype=Vr.dtype)
                    sum_rank = 0
                    for i in range(num_experts):
                        ri = rank_list[i]
                        scale_i = self.scaling[adapter_name][i]
                        Sr_i = S[sum_rank:sum_rank + ri] / (scale_i * rho)
                        sqrt_Sr_i = torch.sqrt(Sr_i)
                        lora_A_i = torch.diag(sqrt_Sr_i) @ Uhr[sum_rank:sum_rank + ri, :]
                        lora_B_i = Vr[:, sum_rank:sum_rank + ri] @ torch.diag(sqrt_Sr_i)
                        lora_A[sum_rank:sum_rank + ri, :] = lora_A_i
                        lora_B[:, sum_rank:sum_rank + ri] = lora_B_i
                        self.lora_A[adapter_name][i].weight.data = (
                            lora_A_i.contiguous())
                        self.lora_B[adapter_name][i].weight.data = (
                            lora_B_i.contiguous())
                        sum_rank += ri
                    residual = torch.zeros_like(weight.data)
                    sum_rank = 0
                    for i, ri in enumerate(rank_list):
                        lora_A_i = lora_A[sum_rank:sum_rank + ri, :]
                        lora_B_i = lora_B[:, sum_rank:sum_rank + ri]
                        residual += self.scaling[adapter_name][i] * (lora_B_i @ lora_A_i)
                        sum_rank += ri
                    self.get_base_layer().weight.data -= init_cof * residual
                else:
                    scaling = self.scaling[adapter_name][0]
                    Sr = S / (scaling * rho)
                    sqrt_Sr = torch.sqrt(Sr)
                    lora_A = torch.diag(sqrt_Sr) @ Uhr
                    lora_B = Vr @ torch.diag(sqrt_Sr)
                    sum_rank = 0
                    for i in range(num_experts):
                        ri = rank_list[i]
                        self.lora_A[adapter_name][i].weight.data = (
                            lora_A[sum_rank:sum_rank + ri, :].contiguous())
                        self.lora_B[adapter_name][i].weight.data = (
                            lora_B[:, sum_rank:sum_rank + ri].contiguous())
                        sum_rank += ri
                    self.get_base_layer().weight.data -= (
                        init_cof * scaling * lora_B @ lora_A)
            svd_init()
        else:
            # Default: kaiming initialization
            for i in range(len(self.lora_A[adapter_name])):
                nn.init.kaiming_uniform_(self.lora_A[adapter_name][i].weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B[adapter_name][i].weight)

        # Create generalized experts (always activated)
        generalized_experts = nn.ModuleList(GSEExpert(
                self.lora_A[adapter_name][i],
                self.lora_B[adapter_name][i],
                self.lora_dropout[adapter_name][i],
                self.scaling[adapter_name][i],
        ) for i in range(num_generalized_experts))
        
        # Create specialized experts (selected by router)
        specialized_experts = nn.ModuleList(GSEExpert(
                self.lora_A[adapter_name][num_generalized_experts + i],
                self.lora_B[adapter_name][num_generalized_experts + i],
                self.lora_dropout[adapter_name][num_generalized_experts + i],
                self.scaling[adapter_name][num_generalized_experts + i],
        ) for i in range(num_specialized_experts))
        
        self.moe_layer[adapter_name] = TopKGSELayer(
            generalized_experts=generalized_experts,
            specialized_experts=specialized_experts,
            gate=self.lora_gating[adapter_name],
            top_k=top_k,
            compute_aux_loss=(aux_loss_weight > 0),
        )
        self.set_adapter(self.active_adapters)

    def reset_parameters(self, adapter_name: str, init_lora_weights: bool) -> None:
        if init_lora_weights is False:
            return
        elif adapter_name in self.lora_A.keys():
            for i in range(len(self.lora_A[adapter_name])):
                nn.init.kaiming_uniform_(self.lora_A[adapter_name][i].weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B[adapter_name][i].weight)


class LinearGSELayer(nn.Module, GSELayer):
    """Linear layer with GSE adaptation"""
    
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        lora_rank: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: bool = True,
        num_experts: int = 8,
        num_generalized_experts: int = 2,
        top_k: int = 2,
        init_type: str = "gse",
        init_cof: float = 1.0,
        specialized_scaling_method: str = "default",
        specialized_scaling_base: float = 2.0,
        specialized_scaling_eps: float = 1e-12,
        skip_svd_init: bool = False,
        aux_loss_weight: float = 0.01,
        **kwargs,
    ) -> None:
        super().__init__()
        GSELayer.__init__(self, base_layer=base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name, lora_rank, lora_alpha, lora_dropout, init_lora_weights,
            num_experts, num_generalized_experts, top_k, init_type, init_cof,
            specialized_scaling_method=specialized_scaling_method,
            specialized_scaling_base=specialized_scaling_base,
            specialized_scaling_eps=specialized_scaling_eps,
            skip_svd_init=skip_svd_init, aux_loss_weight=aux_loss_weight)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype
        result = self.base_layer(x, *args, **kwargs)
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            moe_layer = self.moe_layer[active_adapter]
            # Use dtype from generalized experts if available, else specialized
            if len(moe_layer.generalized_experts) > 0:
                x = x.to(moe_layer.generalized_experts[0].lora_A.weight.dtype)
            elif len(moe_layer.specialized_experts) > 0:
                x = x.to(moe_layer.specialized_experts[0].lora_A.weight.dtype)
            result += moe_layer(x)

        result = result.to(previous_dtype)
        return result
