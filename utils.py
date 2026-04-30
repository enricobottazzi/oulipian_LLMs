def decode_stego(new_ids: list[int], stride: int, n_bits: int) -> list[int]:
    "Recover bits from generated token ids assuming the same `stride`. Returns up to n_bits."
    return [new_ids[i] & 1 for i in range(stride - 1, len(new_ids), stride)][:n_bits]
