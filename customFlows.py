from nflows.transforms.autoregressive import AutoregressiveTransform
from nflows.transforms.base import Transform

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from nflows.transforms import made as made_module
from nflows.transforms.splines.cubic import cubic_spline
from nflows.transforms.splines.linear import linear_spline
from nflows.transforms.splines.quadratic import (
    quadratic_spline,
    unconstrained_quadratic_spline,
)
from nflows.transforms.splines import rational_quadratic
from nflows.transforms.splines.rational_quadratic import rational_quadratic_spline,unconstrained_rational_quadratic_spline
from nflows.utils import torchutils

class ResidualBlock(nn.Module):
    """A residual block containing masked linear modules."""

    def __init__(self,dim,activation=F.relu,):
        super().__init__()

        # Masked linear.
        linear_0 = nn.Linear(in_features=dim,out_features=dim)
        linear_1 = nn.Linear(in_features=dim,out_features=dim)
        self.linear_layers = nn.ModuleList([linear_0, linear_1])
        
        # Activation and dropout
        self.activation = activation

    def forward(self, inputs):
        temps = inputs
        temps = self.linear_layers[0](temps)
        temps = self.activation(temps)
        temps = self.linear_layers[1](temps)
        return self.activation(inputs + temps)

class ResidualNN(nn.Module):
    def __init__(self,input_dim,hidden_features,output_dim,activation=F.relu,num_blocks=2):
        super().__init__()
        self.layers = []
        self.act = activation
        self.first = nn.Linear(in_features=input_dim,out_features=hidden_features)
        for i in range(num_blocks):
            self.layers.append(
                ResidualBlock(hidden_features,activation=self.act)
            )
        self.layers = nn.ModuleList(self.layers)
        self.last = nn.Linear(in_features=hidden_features,out_features=output_dim)
        
    def forward(self,inputs,context):
        output = self.act(self.first(context))
        for layer in self.layers:
            output = layer(output)
        output = self.last(output)
        return output
    
class RQSparams1D(nn.Module):
    def __init__(self,features,multiplier,hidden,n_hidden,context=None,dropout=0.0,residual=False):
        super().__init__()
        #self.mat = torch.zeros(multiplier*features,features).float()
        self.dropout = dropout
        n_in = features if context is None else context
        out_dim = multiplier*features
        modules = [nn.Linear(n_in,hidden),nn.ReLU()]
        if self.dropout>0:
            #print(f'applying {self.dropout} dropout')
            modules.append(nn.Dropout(self.dropout))
        for _ in range(n_hidden-1):
            if residual:
                modules.append(ResidualBlock(hidden))
            else:
                modules.append(nn.Linear(hidden,hidden))
            modules.append(nn.ReLU())
            if self.dropout>0:
                modules.append(nn.Dropout(self.dropout))
        modules.append(nn.Linear(hidden,out_dim))
        self.transform = nn.Sequential(*modules)
    
    def forward(self,inputs,context=None):
        if context is not None:
            return self.transform(context)
        else:
            return self.transform((inputs*0.0).float())
    
class RQSparamsND(nn.Module):
    def __init__(self,features,multiplier,hidden,n_hidden,dropout=0.0,residual=False):
        super().__init__()
        nets = [NeuralNet(1,hidden,multiplier,1,out_act=nn.Identity())]
        for i in range(1,features):
            cont = i if i>0 else None
            nets.append(RQSparams1D(1,multiplier,hidden,n_hidden,context=cont,
                               dropout=dropout,residual=residual))
        self.nets = nn.ModuleList(nets)
    
    def forward(self,inputs,context=None):
        #return inputs@self.mat.T + self.params
        params = [self.nets[0](0.0*inputs[:,:1])]
        for i in range(1,inputs.shape[1]):
            inp = inputs[:,i].view(-1,1)
            context = None if i==0 else inputs[:,:i]
            params.append(self.nets[i](inputs=inp,context=context))
        return torch.cat(params,dim=1)

class NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers,act=None,out_act=None):
        super().__init__()
        layers = []
        dcurr = input_dim
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(dcurr, hidden_dim))
            layers.append(nn.ReLU() if act is None else act)
            dcurr = hidden_dim
        layers.append(nn.Linear(dcurr, output_dim))
        layers.append(nn.Tanh() if out_act is None else out_act)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)   

class IndependentRQS(Transform):
    def __init__(
        self,
        features,
        context=None,
        hidden=50,
        num_hidden=1,
        num_bins=10,
        tails=None,
        tail_bound=1.0,
        dropout=0.0,
        residual=False,
        min_bin_width=rational_quadratic.DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=rational_quadratic.DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=rational_quadratic.DEFAULT_MIN_DERIVATIVE
    ):
        super().__init__()
        self.features = features
        self.context = context
        self.hidden=hidden
        self.num_hidden = num_hidden
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.tails = tails
        self.tail_bound = tail_bound
        self.dropout = dropout
        self.residual = residual
        self.net = RQSparams1D(features,self._output_dim_multiplier(),self.hidden,self.num_hidden,context=self.context,
                               dropout=self.dropout,residual=self.residual)
        #self.params = nn.Parameter(nn.init.uniform_(torch.empty(self.features*self._output_dim_multiplier())),requires_grad=True)
        #self.params = nn.Parameter(torch.randn(self.features*self._output_dim_multiplier()),requires_grad=True)
                
    def _output_dim_multiplier(self):
        if self.tails == "linear" or self.tails == "multi_linear":
            return self.num_bins * 3 - 1
        elif self.tails is None:
            return self.num_bins * 3 + 1
        else:
            raise ValueError
            
    def forward(self, inputs, context=None):
        #params = self.params.tile((inputs.shape[0],1))
        params = self.net(inputs,context=context)
        outputs, logabsdet = self._elementwise_forward(inputs, params)
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        #params = self.params.tile((inputs.shape[0],1))
        params = self.net(inputs,context=context)
        outputs, logabsdet = self._elementwise_inverse(inputs, params)
        return outputs, logabsdet
    
    def _elementwise(self, inputs, autoregressive_params, inverse=False):
        batch_size = inputs.shape[0]
        features = inputs.shape[1]
        if features != self.features:
            raise InputError('Input dimensionality does not match that of model')

        transform_params = autoregressive_params.view(
            batch_size, features, self._output_dim_multiplier()
        )

        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = transform_params[..., 2 * self.num_bins :]

        if self.tails is None:
            spline_fn = rational_quadratic_spline
            if type(self.tail_bound)==list:
                spline_kwargs = {'left':self.tail_bound[0],'right':self.tail_bound[1],
                                 'bottom':self.tail_bound[2],'top':self.tail_bound[3]}
            else:
                spline_kwargs = {}
        elif self.tails == "linear":
            spline_fn = unconstrained_rational_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}
        elif self.tails == "multi_linear":
            spline_fn = unconstrained_rational_quadratic_spline_multi_bounds
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}
        else:
            raise ValueError
            
        if self.tails is None:
            constant = np.log(np.exp(1 - self.min_derivative) - 1)
            unnormalized_derivatives[..., 0] = constant
            unnormalized_derivatives[..., -1] = constant

        outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs
        )

        return outputs, torchutils.sum_except_batch(logabsdet)

    def _elementwise_forward(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params)

    def _elementwise_inverse(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params, inverse=True) 

class AutoregressiveRQS(Transform):
    def __init__(
        self,
        features,
        context=None,
        hidden=50,
        num_hidden=1,
        num_bins=10,
        tails=None,
        tail_bound=1.0,
        dropout=0.0,
        residual=False,
        min_bin_width=rational_quadratic.DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=rational_quadratic.DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=rational_quadratic.DEFAULT_MIN_DERIVATIVE
    ):
        super().__init__()
        self.features = features
        self.context = context
        self.hidden=hidden
        self.num_hidden = num_hidden
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.tails = tails
        self.tail_bound = tail_bound
        self.dropout = dropout
        self.residual = residual
        self.net = RQSparamsND(features,self._output_dim_multiplier(),self.hidden,self.num_hidden,
                               dropout=self.dropout,residual=self.residual)
        #self.p1 = nn.Parameter(nn.init.uniform_(torch.empty(self._output_dim_multiplier())),requires_grad=True)
        #self.params = nn.Parameter(nn.init.uniform_(torch.empty(self.features*self._output_dim_multiplier())),requires_grad=True)
        #self.params = nn.Parameter(torch.randn(self.features*self._output_dim_multiplier()),requires_grad=True)
                
    def _output_dim_multiplier(self):
        if self.tails == "linear" or self.tails == "multi_linear":
            return self.num_bins * 3 - 1
        elif self.tails is None:
            return self.num_bins * 3 + 1
        else:
            raise ValueError
            
    def forward(self, inputs, context=None):
        #params = self.params.tile((inputs.shape[0],1))
        params = self.net(inputs,context=context)
        outputs, logabsdet = self._elementwise_forward(inputs, params)
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        #params = self.params.tile((inputs.shape[0],1))
        params = self.net(inputs,context=context)
        outputs, logabsdet = self._elementwise_inverse(inputs, params)
        return outputs, logabsdet
    
    def _elementwise(self, inputs, autoregressive_params, inverse=False):
        batch_size = inputs.shape[0]
        features = inputs.shape[1]
        if features != self.features:
            raise InputError('Input dimensionality does not match that of model')

        transform_params = autoregressive_params.view(
            batch_size, features, self._output_dim_multiplier()
        )

        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = transform_params[..., 2 * self.num_bins :]

        if self.tails is None:
            spline_fn = rational_quadratic_spline
            if type(self.tail_bound)==list:
                spline_kwargs = {'left':self.tail_bound[0],'right':self.tail_bound[1],
                                 'bottom':self.tail_bound[2],'top':self.tail_bound[3]}
            else:
                spline_kwargs = {}
        elif self.tails == "linear":
            spline_fn = unconstrained_rational_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}
        elif self.tails == "multi_linear":
            spline_fn = unconstrained_rational_quadratic_spline_multi_bounds
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}
        else:
            raise ValueError
            
        if self.tails is None:
            constant = np.log(np.exp(1 - self.min_derivative) - 1)
            unnormalized_derivatives[..., 0] = constant
            unnormalized_derivatives[..., -1] = constant

        outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs
        )

        return outputs, torchutils.sum_except_batch(logabsdet)

    def _elementwise_forward(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params)

    def _elementwise_inverse(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params, inverse=True)

    
class ConditionalRationalQuadraticAutoregressiveTransform(AutoregressiveTransform):
    def __init__(
        self,
        features,
        context_features,
        hidden_features=128,
        num_bins=10,
        tails=None,
        tail_bound=1.0,
        num_blocks=2,
        use_residual_blocks=True,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        min_bin_width=rational_quadratic.DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=rational_quadratic.DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=rational_quadratic.DEFAULT_MIN_DERIVATIVE,
    ):
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.tails = tails
        self.tail_bound = tail_bound
        self.num_blocks = num_blocks

        autoregressive_net = ResidualNN(
            context_features,
            hidden_features,
            features*self._output_dim_multiplier(),
            num_blocks = self.num_blocks
        )
        
        super().__init__(autoregressive_net)

    def _output_dim_multiplier(self):
        if self.tails == "linear" or self.tails == "multi_linear":
            return self.num_bins * 3 - 1
        elif self.tails is None:
            return self.num_bins * 3 + 1
        else:
            raise ValueError

    def _elementwise(self, inputs, autoregressive_params, inverse=False):
        batch_size, features = inputs.shape[0], inputs.shape[1]

        transform_params = autoregressive_params.view(
            batch_size, features, self._output_dim_multiplier()
        )

        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = transform_params[..., 2 * self.num_bins :]

        if hasattr(self.autoregressive_net, "hidden_features"):
            unnormalized_widths /= np.sqrt(self.autoregressive_net.hidden_features)
            unnormalized_heights /= np.sqrt(self.autoregressive_net.hidden_features)

        if self.tails is None:
            spline_fn = rational_quadratic_spline
            if type(self.tail_bound)==list:
                spline_kwargs = {'left':self.tail_bound[0],'right':self.tail_bound[1],
                                 'bottom':self.tail_bound[2],'top':self.tail_bound[3]}
            else:
                spline_kwargs = {}
        elif self.tails == "linear":
            spline_fn = unconstrained_rational_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}
        elif self.tails == "multi_linear":
            spline_fn = unconstrained_rational_quadratic_spline_multi_bounds
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}
        else:
            raise ValueError
            
        if self.tails is None:
            constant = np.log(np.exp(1 - self.min_derivative) - 1)
            unnormalized_derivatives[..., 0] = constant
            unnormalized_derivatives[..., -1] = constant

        outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs
        )

        return outputs, torchutils.sum_except_batch(logabsdet)

    def _elementwise_forward(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params)

    def _elementwise_inverse(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params, inverse=True)
    
    
    
class RQSBase(Transform):
    def __init__(self,num_bins,tail_bound,tails,min_bin_width=rational_quadratic.DEFAULT_MIN_BIN_WIDTH,
                 min_bin_height=rational_quadratic.DEFAULT_MIN_BIN_HEIGHT,min_derivative=rational_quadratic.DEFAULT_MIN_DERIVATIVE):
        super().__init__()
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.tails = tails
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        
    def get_params(self,inputs,context=None):
        raise NotImplementedError()
        
    def forward(self, inputs, context=None):
        params = self.get_params(inputs, context)
        outputs, logabsdet = self._elementwise_forward(inputs, params)
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        num_inputs = np.prod(inputs.shape[1:])
        outputs = torch.zeros_like(inputs)
        logabsdet = None
        for _ in range(num_inputs):
            params = self.get_params(outputs, context)
            outputs, logabsdet = self._elementwise_inverse(
                inputs, params
            )
        return outputs, logabsdet

    def _output_dim_multiplier(self):
        if self.tails == "linear" or self.tails == "multi_linear":
            return self.num_bins * 3 - 1
        elif self.tails is None:
            return self.num_bins * 3 + 1
        else:
            raise ValueError

    def _elementwise(self, inputs, params, inverse=False):
        batch_size, features = inputs.shape[0], inputs.shape[1]

        transform_params = params.view(
            batch_size, features, self._output_dim_multiplier()
        )

        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = transform_params[..., 2 * self.num_bins :]

        if hasattr(self, "hidden"):
            unnormalized_widths /= np.sqrt(self.hidden)
            unnormalized_heights /= np.sqrt(self.hidden)

        if self.tails is None:
            spline_fn = rational_quadratic_spline
            if type(self.tail_bound)==list:
                spline_kwargs = {'left':self.tail_bound[0],'right':self.tail_bound[1],
                                 'bottom':self.tail_bound[2],'top':self.tail_bound[3]}
            else:
                spline_kwargs = {}
        elif self.tails == "linear":
            spline_fn = unconstrained_rational_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}
        elif self.tails == "multi_linear":
            spline_fn = unconstrained_rational_quadratic_spline_multi_bounds
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}
        else:
            raise ValueError

        outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs
        )

        return outputs, torchutils.sum_except_batch(logabsdet)

    def _elementwise_forward(self, inputs, params):
        return self._elementwise(inputs, params)

    def _elementwise_inverse(self, inputs, params):
        return self._elementwise(inputs, params, inverse=True)
    
class Conditional1DRQS(RQSBase):
    def __init__(
        self,
        features,
        context=None,
        hidden=50,
        num_hidden=1,
        num_bins=10,
        tails="linear",
        tail_bound=1.0,
        dropout=0.0,
        residual=False,
        useNet_noContext=True
    ):
        super().__init__(num_bins,tail_bound,tails)
        assert features == 1
        self.features = features
        self.context = context
        self.hidden = hidden
        self.num_hidden = num_hidden
        self.dropout = dropout
        self.residual = False
        self.useNet_noContext = useNet_noContext
        if self.context is None and not self.useNet_noContext:
            self.param_net = nn.Parameter(nn.init.uniform_(torch.empty(self._output_dim_multiplier())),requires_grad=True)
        else:
            self.param_net = RQSparams1D(self.features,self._output_dim_multiplier(),self.hidden,self.num_hidden,context=self.context,
                               dropout=self.dropout,residual=self.residual)
        
    def get_params(self,inputs,context=None):
        if context is None and not self.useNet_noContext:
            return self.param_net.repeat(inputs.shape[0],1)
        else:
            return self.param_net(inputs,context=context)
        
class ConditionalMultiRQS(RQSBase):
    def __init__(
        self,
        features,
        num_context=0,
        hidden=50,
        num_hidden=1,
        num_bins=10,
        tails="linear",
        tail_bound=1.0,
        dropout=0.0,
        residual=False,
    ):
        super().__init__(num_bins,tail_bound,tails)
        self.features = features
        self.num_context = num_context if num_context is not None else 0
        self.hidden = hidden
        self.num_hidden = num_hidden
        self.dropout = dropout
        self.residual = False
        self.nets = []
        for i in range(self.features):
            nc = i+self.num_context
            self.nets.append(RQSparams1D(1,self._output_dim_multiplier(),self.hidden,self.num_hidden,context=nc if nc>0 else None,
                               dropout=self.dropout,residual=self.residual))
        self.nets = nn.ModuleList(self.nets)
        
    def get_params(self,inputs,context=None):
        params = []
        for i in range(self.features):
            if context is not None:
                cont = torch.cat((context,inputs[:,:i]),dim=-1)
            else:
                cont = None if i==0 else inputs[:,:i]
            params.append(self.nets[i](inputs[:,:i+1],context=cont))
        return torch.cat(params,dim=-1)