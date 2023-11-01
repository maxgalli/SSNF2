from nflows.flows.base import Flow
from nflows.distributions import ConditionalDiagonalNormal
from nflows import transforms
from ffflows.distance_penalties import BasePenalty
from ffflows.distance_penalties import AnnealedPenalty
from ffflows import distance_penalties

import zuko
from zuko.flows import NSF

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path


def set_penalty(f4flow, penalty, weight, anneal=False):
    if penalty not in ["None", None]:
        if penalty == "l1":
            penalty_constr = distance_penalties.LOnePenalty
        elif penalty == "l2":
            penalty_constr = distance_penalties.LTwoPenalty
        penalty = penalty_constr(weight)
        if anneal:
            penalty = AnnealedPenalty(penalty)
        f4flow.add_penalty(penalty)


def spline_inn(
    inp_dim,
    nodes=128,
    num_blocks=2,
    num_stack=3,
    tail_bound=3.5,
    tails="linear",
    activation=F.relu,
    lu=0,
    num_bins=12,
    context_features=None,
    dropout_probability=0.0,
    flow_for_flow=False,
):
    transform_list = []
    for i in range(num_stack):
        transform_list += [
            transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                inp_dim,
                nodes,
                num_blocks=num_blocks,
                tail_bound=tail_bound,
                num_bins=num_bins,
                tails=tails,
                activation=activation,
                dropout_probability=dropout_probability,
                context_features=context_features,
            )
        ]
        if lu:
            transform_list += [transforms.LULinear(inp_dim)]
        else:
            transform_list += [transforms.ReversePermutation(inp_dim)]

    if not (flow_for_flow and (num_stack % 2 == 0)):
        # If the above conditions are satisfied then you want to permute back to the original ordering such that the
        # output features line up with their original ordering.
        transform_list = transform_list[:-1]

    return transforms.CompositeTransform(transform_list)


def get_conditional_base_flow(
    input_dim,
    context_dim,
    nstack,
    nnodes,
    nblocks,
    tail_bound,
    nbins,
    activation,
    dropout_probability,
):
    flow = Flow(
        spline_inn(
            input_dim,
            nodes=nnodes,
            num_blocks=nblocks,
            num_stack=nstack,
            tail_bound=tail_bound,
            activation=getattr(F, activation),
            dropout_probability=dropout_probability,
            num_bins=nbins,
            context_features=context_dim,
        ),
        ConditionalDiagonalNormal(
            shape=[input_dim], context_encoder=nn.Linear(context_dim, 2 * input_dim)
        ),
    )

    flow.model_hyperparams = {
        "input_dim": input_dim,
        "context_dim": context_dim,
        "nstack": nstack,
        "nnodes": nnodes,
        "nblocks": nblocks,
        "tail_bound": tail_bound,
        "nbins": nbins,
        "activation": activation,
        "dropout_probability": dropout_probability,
    } 

    return flow

    
def get_zuko_nsf(
    input_dim,
    context_dim,
    ntransforms,
    nbins,
    nnodes,
    nlayers,
    mc_flow=None,
    data_flow=None,
    penalty=None,
):
    if data_flow is None and mc_flow is None and penalty is None:
        flow = NSF(
            features=input_dim,
            context=context_dim,
            transforms=ntransforms,
            bins=nbins,
            hidden_features=[nnodes] * nlayers,
        )
    else:
        flow = FFFZuko(
            NSF(
                features=input_dim,
                context=context_dim,
                transforms=ntransforms,
                bins=nbins,
                hidden_features=[nnodes] * nlayers,
            ),
            mc_flow,
            data_flow,
        )
        set_penalty(flow, penalty["penalty_type"], penalty["penalty_weight"], penalty["anneal"])

    flow.model_hyperparams = {
        "input_dim": input_dim,
        "context_dim": context_dim,
        "ntransforms": ntransforms,
        "nbins": nbins,
        "nnodes": nnodes,
        "nlayers": nlayers,
    }

    return flow

    
class FFFZuko(zuko.flows.core.Flow):
    def __init__(self, transform, flow_mc, flow_data):
        super().__init__(transform.transforms, transform.base)
        self._transform = transform
        self.flow_mc = flow_mc
        self.flow_data = flow_data

    def add_penalty(self, penalty_object):
        """Add a distance penaly object to the class."""
        assert isinstance(penalty_object, BasePenalty)
        self.distance_object = penalty_object

    def base_flow_log_prob(
        self, inputs, context, inverse=False
    ):
        if inverse:
            fnc = self.flow_mc(context).log_prob
        else:
            fnc = self.flow_data(context).log_prob
        logprob = fnc(inputs)
        return logprob

    def transform(self, inputs, context, inverse=False):
        transform = self._transform(context).transform.inv if inverse else self._transform(context).transform
        y = transform(inputs)
        logabsdet = transform.log_abs_det_jacobian(inputs, inputs)

        return y, logabsdet

    def log_prob(self, inputs, context, inverse=False):
        converted_input, logabsdet = self.transform(
            inputs, context, inverse=inverse
        )
        log_prob = self.base_flow_log_prob(
            converted_input, context, inverse=inverse
        )
        dist_pen = self.distance_object(converted_input, inputs)

        return log_prob, logabsdet, dist_pen


def save_model(
    epoch,
    model,
    scheduler,
    train_history,
    test_history,
    name,
    model_dir=None,
    optimizer=None,
    is_ddp=False,
):
    """Save a model and optimizer to file.
    Args:
        model:      model to be saved
        optimizer:  optimizer to be saved
        epoch:      current epoch number
        model_dir:  directory to save the model in
        filename:   filename for saved model
    """
    if model_dir is None:
        raise NameError("Model directory must be specified.")

    filename = name

    p = Path(model_dir)
    p.mkdir(parents=True, exist_ok=True)

    dict = {
        "train_history": train_history,
        "test_history": test_history,
        "model_hyperparams": model.module.model_hyperparams
        if is_ddp
        else model.model_hyperparams,
        "model_state_dict": model.module.state_dict() if is_ddp else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }

    if scheduler is not None:
        dict["scheduler_state_dict"] = scheduler.state_dict()
        dict["last_lr"] = scheduler.get_last_lr()

    torch.save(dict, p / filename)


def load_model(device, model_dir=None, filename=None, which="zuko_nfs"):
    """Load a saved model.
    Args:
        filename:       File name
    """
    if which == "zuko_nsf":
        create_function = get_zuko_nsf
    else:
        raise ValueError("which must be zuko")

    if model_dir is None:
        raise NameError(
            "Model directory must be specified."
            " Store in attribute PosteriorModel.model_dir"
        )

    p = Path(model_dir)
    checkpoint = torch.load(p / filename, map_location="cpu")

    model_hyperparams = checkpoint["model_hyperparams"]
    # added because of a bug in the old create_mixture_flow_model function
    try:
        if checkpoint["model_hyperparams"]["base_transform_kwargs"] is not None:
            checkpoint["model_hyperparams"]["base_kwargs"] = checkpoint[
                "model_hyperparams"
            ]["base_transform_kwargs"]
            del checkpoint["model_hyperparams"]["base_transform_kwargs"]
    except KeyError:
        pass
    train_history = checkpoint["train_history"]
    test_history = checkpoint["test_history"]

    # Load model
    model = create_function(**model_hyperparams)
    model.load_state_dict(checkpoint["model_state_dict"])
    # model.to(device)

    # Remember that you must call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference.
    # Failing to do this will yield inconsistent inference results.
    model.eval()

    # Load optimizer
    scheduler_present_in_checkpoint = "scheduler_state_dict" in checkpoint.keys()

    # If the optimizer has more than 1 param_group, then we built it with
    # flow_lr different from lr
    if len(checkpoint["optimizer_state_dict"]["param_groups"]) > 1:
        flow_lr = checkpoint["last_lr"]
    elif checkpoint["last_lr"] is not None:
        flow_lr = checkpoint["last_lr"][0]
    else:
        flow_lr = None

    # Set the epoch to the correct value. This is needed to resume
    # training.
    epoch = checkpoint["epoch"]

    return (
        model,
        scheduler_present_in_checkpoint,
        flow_lr,
        epoch,
        train_history,
        test_history,
    )


def load_fff_model(top_file, mc_file, data_file, top_penalty, which="zuko_nsf"):
    if which == "zuko_nsf":
        create_fn = get_zuko_nsf
    checkpoint_top = torch.load(top_file, map_location="cpu")
    checkpoint_mc = torch.load(mc_file, map_location="cpu")
    checkpoint_data = torch.load(data_file, map_location="cpu")

    model_mc = create_fn(**checkpoint_mc["model_hyperparams"])
    # model_mc.load_state_dict(checkpoint_mc["model_state_dict"])
    # model_mc.eval()
    model_data = create_fn(**checkpoint_data["model_hyperparams"])
    # model_data.load_state_dict(checkpoint_data["model_state_dict"])
    # model_data.eval()

    model_top = create_fn(
        **checkpoint_top["model_hyperparams"],
        mc_flow=model_mc,
        data_flow=model_data,
        penalty=top_penalty,
    )
    # print(model_top.state_dict()["_distribution._transform._transforms.30.autoregressive_net.blocks.0.linear_layers.0.weight"])
    model_top.load_state_dict(checkpoint_top["model_state_dict"])
    # print(model_top.state_dict()["_distribution._transform._transforms.30.autoregressive_net.blocks.0.linear_layers.0.weight"])
    model_top.eval()

    scheduler_present_in_checkpoint = "scheduler_state_dict" in checkpoint_top.keys()

    if len(checkpoint_top["optimizer_state_dict"]["param_groups"]) > 1:
        flow_lr = checkpoint_top["last_lr"]
    elif checkpoint_top["last_lr"] is not None:
        flow_lr = checkpoint_top["last_lr"][0]
    else:
        flow_lr = None

    epoch = checkpoint_top["epoch"]
    train_history = checkpoint_top["train_history"]
    test_history = checkpoint_top["test_history"]

    return (
        model_top,
        scheduler_present_in_checkpoint,
        flow_lr,
        epoch,
        train_history,
        test_history,
    )