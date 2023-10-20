from nflows.flows.base import Flow
from nflows.distributions import ConditionalDiagonalNormal
from nflows import transforms
from ffflows.distance_penalties import BasePenalty
from ffflows.distance_penalties import AnnealedPenalty
from ffflows import distance_penalties

import zuko
from zuko.flows import NSF

import torch.nn as nn
import torch.nn.functional as F


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
        dist_pen = -self.distance_object(converted_input, inputs)

        return log_prob, logabsdet, dist_pen