import numpy as np
import pytest
from mindset.acl.clients import prompt_tokens


class Tokenizer:
    def __init__(self,result): self.result=result; self.kwargs=None
    def apply_chat_template(self,messages,**kwargs): self.kwargs=kwargs; return self.result


@pytest.mark.parametrize('result',[[1,2,3,4],{'input_ids':[1,2,3,4],'attention_mask':[1,1,1,1]},
    {'input_ids':[[1,2,3,4]],'attention_mask':[[1,1,1,1]]},np.asarray([1,2,3,4]),
    {'input_ids':np.asarray([[1,2,3,4]])}])
def test_chat_token_count_handles_flat_batched_and_mapping_results(result):
    tokenizer=Tokenizer(result)
    assert prompt_tokens(tokenizer,[{'role':'user','content':'hello'}],thinking=False)==4
    assert tokenizer.kwargs=={'tokenize':True,'add_generation_prompt':True,'enable_thinking':False}


@pytest.mark.parametrize('result',[{'attention_mask':[1,1]},[[1,2],[3,4]],['not an ID'],{'input_ids':'bad'},None])
def test_invalid_chat_tokenization_is_rejected(result):
    with pytest.raises(ValueError): prompt_tokens(Tokenizer(result),[])


def test_empty_prompt_tokens_count_zero_and_thinking_option_is_not_invented():
    tokenizer=Tokenizer([])
    assert prompt_tokens(tokenizer,[])==0
    assert 'enable_thinking' not in tokenizer.kwargs
