import numpy as np
import torch


def fp8_e4m3_to_float(raw):
    raw = raw.astype(np.uint8)
    sign = np.where((raw & 0x80) != 0, -1.0, 1.0).astype(np.float32)
    exp = ((raw >> 3) & 0x0F).astype(np.int32)
    mant = (raw & 0x07).astype(np.float32)
    sub = (mant / 8.0) * np.power(2.0, -6, dtype=np.float32)
    norm = (1.0 + mant / 8.0) * np.power(2.0, (exp - 7).astype(np.float32))
    value = np.where(exp == 0, sub, norm).astype(np.float32) * sign
    return np.where((raw & 0x7F) == 0, 0.0, value).astype(np.float32)


def dequantize_e4m3(tensor, sf, block_size):
    values = fp8_e4m3_to_float(tensor.view(torch.uint8).detach().numpy())
    scale = sf.detach().numpy()
    scale = np.repeat(scale, block_size[0], axis=0)
    scale = np.repeat(scale, block_size[1], axis=1)
    scale = scale[: values.shape[0], : values.shape[1]]
    return values * scale


def assert_dequantized_close(actual, actual_sf, expected, expected_sf, block_size, *, atol=1e-2, rtol=1e-2):
    actual_values = dequantize_e4m3(actual, actual_sf, block_size)
    expected_values = dequantize_e4m3(expected, expected_sf, block_size)
    diff = np.abs(actual_values - expected_values)
    limit = atol + rtol * np.abs(expected_values)
    max_over = float(np.max(diff - limit))
    assert bool(np.all(diff <= limit)), f"max_over={max_over}, max_abs={float(np.max(diff))}"


def fp4_e2m1_to_float(raw):
    raw = raw.astype(np.uint8) & 0x0F
    sign = np.where((raw & 0x08) != 0, -1.0, 1.0).astype(np.float32)
    exp = ((raw >> 1) & 0x03).astype(np.int32)
    mant = (raw & 0x01).astype(np.float32)
    sub = mant * 0.5
    norm = (1.0 + mant * 0.5) * np.power(2.0, (exp - 1).astype(np.float32))
    return (np.where(exp == 0, sub, norm).astype(np.float32) * sign).astype(np.float32)


def unpack_fp4_e2m1_x2(packed):
    packed_u8 = packed.view(torch.uint8).detach().numpy()
    lo = packed_u8 & 0x0F
    hi = (packed_u8 >> 4) & 0x0F
    return np.stack([lo, hi], axis=-1).reshape(*packed_u8.shape[:-1], packed_u8.shape[-1] * 2)


def dequantize_e2m1_unpacked(unpacked, sf, block_size):
    values = fp4_e2m1_to_float(unpacked.view(torch.uint8).detach().numpy())
    scale = sf.detach().numpy()
    scale = np.repeat(scale, block_size[0], axis=0)
    scale = np.repeat(scale, block_size[1], axis=1)
    scale = scale[: values.shape[0], : values.shape[1]]
    return values * scale


def dequantize_e2m1_packed(packed, sf, block_size):
    values = fp4_e2m1_to_float(unpack_fp4_e2m1_x2(packed))
    scale = sf.detach().numpy()
    scale = np.repeat(scale, block_size[0], axis=0)
    scale = np.repeat(scale, block_size[1], axis=1)
    scale = scale[: values.shape[0], : values.shape[1]]
    return values * scale


def assert_fp4_dequantized_close(actual_unpacked, actual_sf, expected_packed, expected_sf, block_size, *, atol=1e-2, rtol=1e-2):
    actual_values = dequantize_e2m1_unpacked(actual_unpacked, actual_sf, block_size)
    expected_values = dequantize_e2m1_packed(expected_packed, expected_sf, block_size)
    diff = np.abs(actual_values - expected_values)
    limit = atol + rtol * np.abs(expected_values)
    max_over = float(np.max(diff - limit))
    assert bool(np.all(diff <= limit)), f"max_over={max_over}, max_abs={float(np.max(diff))}"


def assert_e4m3_matches_e2m1_close(actual, actual_sf, expected_packed, expected_sf, *, actual_block_size, expected_block_size, atol=1e-2, rtol=1e-2):
    actual_values = dequantize_e4m3(actual, actual_sf, actual_block_size)
    expected_values = dequantize_e2m1_packed(expected_packed, expected_sf, expected_block_size)
    diff = np.abs(actual_values - expected_values)
    limit = atol + rtol * np.abs(expected_values)
    max_over = float(np.max(diff - limit))
    assert bool(np.all(diff <= limit)), f"max_over={max_over}, max_abs={float(np.max(diff))}"
