"""
This stores various utility functions that are used in for the models.
"""

def get_attention_mask(input_ids, pad_token_id=0):
        return input_ids.ne(pad_token_id).int()

def get_length(input_ids, pad_token_id=0):
    mask = input_ids.eq(pad_token_id)
    length = (~mask).sum(1, keepdim=True)

    return length, mask

def truncate(input_ids, pad_token_id=0):
    assert len(input_ids.shape) == 2, "input_ids must be a 2d tensor"
    max_length = get_length(input_ids, pad_token_id=pad_token_id)[0].max().item()

    return input_ids[:, :max_length]
