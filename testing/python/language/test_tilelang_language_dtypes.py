import torch

import tilelang.language as T


def test_integer_dtypes_convert_to_torch_dtype():
    assert T.dtype("int16").as_torch() is torch.int16
    assert T.dtype("int32").as_torch() is torch.int32
    assert T.dtype("int64").as_torch() is torch.int64

