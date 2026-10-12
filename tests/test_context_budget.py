from mindset.baselines.long_context import fit_long_context
import pytest
class Tokenizer:
    def apply_chat_template(self,messages,**kwargs):
        return " ".join(m["content"] for m in messages).split()
def test_context_keeps_latest_whole_events_under_total_budget():
    events=[{"text":" ".join(["old"]*60)},{"text":" ".join(["recent"]*10)}]
    kept,text,omitted=fit_long_context("Where?","",events,Tokenizer(),75,4)
    assert omitted==1 and kept==events[1:] and "old" not in text
def test_question_overflow_fails_without_generation():
    with pytest.raises(ValueError,match="Question itself"):
        fit_long_context(" ".join(["question"]*100),"",[],Tokenizer(),10,4)

