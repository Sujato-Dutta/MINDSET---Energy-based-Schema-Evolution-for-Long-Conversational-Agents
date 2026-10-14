from mindset.baselines.long_context import fit_long_context
import pytest
class Tokenizer:
    def apply_chat_template(self,messages,**kwargs):
        return list(range(len(" ".join(m["content"] for m in messages).split())))
class DictionaryTokenizer(Tokenizer):
    def apply_chat_template(self,messages,**kwargs):
        ids=super().apply_chat_template(messages,**kwargs)
        return {"input_ids":[ids],"attention_mask":[[1]*len(ids)]}
@pytest.mark.parametrize('tokenizer_type',[Tokenizer,DictionaryTokenizer])
def test_context_keeps_latest_whole_events_under_total_budget(tokenizer_type):
    events=[{"text":" ".join(["old"]*60)},{"text":" ".join(["recent"]*10)}]
    kept,text,omitted=fit_long_context("Where?","",events,tokenizer_type(),75,4)
    assert omitted==1 and kept==events[1:] and "old" not in text
@pytest.mark.parametrize('tokenizer_type',[Tokenizer,DictionaryTokenizer])
def test_question_overflow_fails_without_generation(tokenizer_type):
    with pytest.raises(ValueError,match="Question itself"):
        fit_long_context(" ".join(["question"]*100),"",[],tokenizer_type(),10,4)

