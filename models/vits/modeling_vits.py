import math
from dataclasses import dataclass
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.file_utils import ModelOutput

import monotonic_align
from .attentions import Encoder
from .commons import (
    sequence_mask,
    rand_slice_segments,
    generate_path,
    init_weights,
    get_padding,
)
from .configuration_vits import VITSConfig
from .modules import (
    Log,
    ElementwiseAffine,
    DDSConv,
    ConvFlow,
    Flip,
    WN,
    LRELU_SLOPE,
    ResBlock1,
    ResidualCouplingLayer,
)

logger = logging.get_logger(__name__)


class VITSTextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.out_channels = config.inter_channels
        self.hidden_channels = config.hidden_channels
        self.filter_channels = config.filter_channels
        self.n_heads = config.n_heads
        self.n_layers = config.n_layers
        self.kernel_size = config.kernel_size
        self.p_dropout = config.p_dropout

        self.emb = nn.Embedding(self.vocab_size, self.hidden_channels)

        self.encoder = Encoder(
            self.hidden_channels,
            self.filter_channels,
            self.n_heads,
            self.n_layers,
            self.kernel_size,
            self.p_dropout,
        )
        self.proj = nn.Conv1d(self.hidden_channels, self.out_channels * 2, 1)

    def forward(self, x, x_lengths):
        x = self.emb(x) * math.sqrt(self.hidden_channels)
        x = torch.transpose(x, 1, -1)
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


class VITSStochasticDurationPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        filter_channels = config.hidden_channels
        self.in_channels = config.hidden_channels
        self.filter_channels = 192
        self.kernel_size = 3
        self.p_dropout = 0.5
        self.n_flows = 4

        self.log_flow = Log()
        self.flows = nn.ModuleList()
        self.flows.append(ElementwiseAffine(2))
        for i in range(self.n_flows):
            self.flows.append(
                ConvFlow(2, filter_channels, self.kernel_size, n_layers=3)
            )
            self.flows.append(Flip())

        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = DDSConv(
            filter_channels, self.kernel_size, n_layers=3, p_dropout=self.p_dropout
        )
        self.post_flows = nn.ModuleList()
        self.post_flows.append(ElementwiseAffine(2))
        for i in range(4):
            self.post_flows.append(
                ConvFlow(2, filter_channels, self.kernel_size, n_layers=3)
            )
            self.post_flows.append(Flip())

        self.pre = nn.Conv1d(self.in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = DDSConv(
            filter_channels, self.kernel_size, n_layers=3, p_dropout=self.p_dropout
        )

    def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
        x = torch.detach(x)
        x = self.pre(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask

        if not reverse:
            flows = self.flows
            assert w is not None

            logdet_tot_q = 0
            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask
            e_q = (
                torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype)
                * x_mask
            )
            z_q = e_q
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q += logdet_q
            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += torch.sum(
                (F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2]
            )
            logq = (
                torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q ** 2)) * x_mask, [1, 2])
                - logdet_tot_q
            )

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)
            for flow in flows:
                z, logdet = flow(z, x_mask, g=x, reverse=reverse)
                logdet_tot = logdet_tot + logdet
            nll = (
                torch.sum(0.5 * (math.log(2 * math.pi) + (z ** 2)) * x_mask, [1, 2])
                - logdet_tot
            )
            return nll + logq  # [b]
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
            z = (
                torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype)
                * noise_scale
            )
            for flow in flows:
                z = flow(z, x_mask, g=x, reverse=reverse)
            z0, z1 = torch.split(z, [1, 1], 1)
            logw = z0
            return logw


class VITSResidualCouplingBlock(nn.Module):
    def __init__(
        self, config,
    ):
        super().__init__()
        self.inter_channels = config.inter_channels
        self.hidden_channels = config.hidden_channels
        self.dilation_rate = 5
        self.n_layers = 1
        self.n_flows = 4

        self.flows = nn.ModuleList()
        for i in range(self.n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    self.inter_channels,
                    self.hidden_channels,
                    self.dilation_rate,
                    self.n_layers,
                    self.n_flows,
                    mean_only=True,
                )
            )
            self.flows.append(Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class VITSPosteriorEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_channels = config.spec_channels
        self.out_channels = config.inter_channels
        self.hidden_channels = config.hidden_channels
        self.kernel_size = 5
        self.dilation_rate = 1
        self.n_layers = 16

        self.pre = nn.Conv1d(self.in_channels, self.hidden_channels, 1)
        self.enc = WN(
            self.hidden_channels, self.kernel_size, self.dilation_rate, self.n_layers,
        )
        self.proj = nn.Conv1d(self.hidden_channels, self.out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class VITSGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.initial_channel = config.inter_channels
        self.resblock_dilation_sizes = config.resblock_dilation_sizes
        self.upsample_rates = config.upsample_rates
        self.upsample_initial_channel = config.upsample_initial_channel
        self.resblock_kernel_sizes = config.resblock_kernel_sizes
        self.upsample_kernel_sizes = config.upsample_kernel_sizes
        self.num_kernels = len(self.resblock_kernel_sizes)
        self.num_upsamples = len(self.upsample_rates)

        self.conv_pre = nn.Conv1d(
            self.initial_channel, self.upsample_initial_channel, 7, 1, padding=3
        )

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(
            zip(self.upsample_rates, self.upsample_kernel_sizes)
        ):
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        self.upsample_initial_channel // (2 ** i),
                        self.upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = self.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(self.resblock_kernel_sizes, self.resblock_dilation_sizes)
            ):
                self.resblocks.append(ResBlock1(ch, k, d))

        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class VITSDiscriminatorP(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super().__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        1024,
                        1024,
                        (kernel_size, 1),
                        1,
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
            ]
        )
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class VITSDiscriminatorS(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv1d(1, 16, 15, 1, padding=7)),
                weight_norm(nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                weight_norm(nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                weight_norm(nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                weight_norm(nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                weight_norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = weight_norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class VITSMultiPeriodDiscriminator(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        periods = [2, 3, 5, 7, 11]

        discs = [VITSDiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [
            VITSDiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods
        ]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class VITSPreTrainedModel(PreTrainedModel):
    config_class = VITSConfig
    base_model_prefix = "vits"

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(0.0, self.config.hidden_channels ** -0.5)
        elif isinstance(module, VITSGenerator):
            module.ups.apply(init_weights)


class VITSModel(VITSPreTrainedModel):
    config_class = VITSConfig
    base_model_prefix = "vits"

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.enc_p = VITSTextEncoder(config)
        self.dec = VITSGenerator(config)
        self.enc_q = VITSPosteriorEncoder(config)
        self.flow = VITSResidualCouplingBlock(config)
        self.dp = VITSStochasticDurationPredictor(config)

        self.post_init()

    def forward(self, x, x_lengths, y, y_lengths):
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)

        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths)
        z_p = self.flow(z, y_mask)

        with torch.no_grad():
            s_p_sq_r = torch.exp(-2 * logs_p)
            neg_cent1 = torch.sum(
                -0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True
            )
            neg_cent2 = torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r)
            neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))
            neg_cent4 = torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True)
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = (
                monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1))
                .unsqueeze(1)
                .detach()
            )

        w = attn.sum(2)
        l_length = self.dp(x, x_mask, w)
        l_length = l_length / torch.sum(x_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_slice, ids_slice = rand_slice_segments(z, y_lengths, self.segment_size)
        o = self.dec(z_slice)
        return (
            o,
            l_length,
            attn,
            ids_slice,
            x_mask,
            y_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
        )


class VITSForWaveformGeneration(VITSPreTrainedModel):
    def __init__(self, config: VITSConfig):
        super().__init__(config)

    def forward(
        self,
        x,
        x_lengths,
        noise_scale=1,
        length_scale=1,
        noise_scale_w=1.0,
        max_len=None,
    ):
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)

        logw = self.dp(x, x_mask, reverse=True, noise_scale=noise_scale_w)

        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, reverse=True)
        o = self.dec((z * y_mask)[:, :, :max_len])
        return o, attn, y_mask, (z, z_p, m_p, logs_p)
