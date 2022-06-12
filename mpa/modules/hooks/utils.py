# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from matplotlib import pyplot as plt


def print_report(df, exp, logger=None):
    df_exp = df[df.exp == exp]
    df_pprint = (
        df_exp.assign(open_layer=lambda ddf: ddf.hook_type.map(lambda x: {"pre": 0, "fwd": 1, "bwd": 2}[x])
                      .rolling(2)
                      .apply(lambda x: list(x)[0] == 0 and list(x)[1] == 0))
        .assign(close_layer=lambda ddf: ddf.hook_type.map(lambda x: {"pre": 0, "fwd":